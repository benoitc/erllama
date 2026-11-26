"""Extract one rank's half of layer 0 from the two published 0731 files."""
import hashlib
import io
import json
import struct
import urllib.request
from pathlib import Path

REV = "f71f23d552d664e523b422157b2befbf74040380"
FILES = {
    "iq2": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
    "q4": "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf",
}

def fetch(file, first, size):
    url = f"https://huggingface.co/antirez/deepseek-v4-gguf/resolve/{REV}/{file}?slice={first}-{size}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{first+size-1}"})
    resp = urllib.request.urlopen(req, timeout=120)
    assert resp.status == 206 and resp.headers["Content-Range"].startswith(f"bytes {first}-{first+size-1}/")
    return resp

def header(data):
    f = io.BytesIO(data)
    def read(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    def string():
        return f.read(read("Q")).decode()
    def value(kind):
        if kind == 8: return string()
        if kind == 9:
            subtype, n = read("I"), read("Q")
            return [value(subtype) for _ in range(n)]
        return read(("B","b","H","h","I","i","f","?",None,None,"Q","q","d")[kind])
    assert f.read(4) == b"GGUF" and read("I") == 3
    nt,nk = read("Q"),read("Q")
    meta = {}
    for _ in range(nk):
        k = string()
        meta[k] = value(read("I"))
    tensors = {}
    for _ in range(nt):
        k = string()
        dims = [read("Q") for _ in range(read("I"))]
        tensors[k] = dims, read("I"), read("Q")
    align = meta.get("general.alignment",32)
    return (f.tell()+align-1)//align*align,tensors,meta

for quant, file in FILES.items():
    with fetch(file,0,8*1024*1024) as r:
        size = int(r.headers["Content-Range"].split("/")[-1])
        data = r.read()
    offset, tensors, meta = header(data)
    assert meta["general.architecture"] == "deepseek4"
    print(quant,meta["deepseek4.swiglu_clamp_exp"][0],flush=True)
    assert meta["deepseek4.swiglu_clamp_exp"][0] == 10
    dims, typ, pos = tensors["blk.0.ffn_gate_tid2eid.weight"]
    assert dims == [6,129280] and typ == 26
    with fetch(file,offset+pos,6*129280*4) as r:
        table = r.read()
    if quant == "iq2":
        Path("routes.bin").write_bytes(table)
        with open("vocab.gguf","wb") as f:
            f.write(data[:offset]); f.truncate(size)
    else:
        assert Path("routes.bin").read_bytes() == table
    for role in ("up","gate"):
        dims, typ, pos = tensors[f"blk.0.ffn_{role}_exps.weight"]
        assert dims == [4096,2048,256] and typ == {"iq2":16,"q4":12}[quant],(dims,typ)
        half = (4096//256)*{"iq2":66,"q4":144}[quant]*1024
        dest = Path(f"{quant}-{role}.bin")
        with fetch(file,offset+pos,half*2*256) as r, dest.open("wb") as f:
            for i in range(256):
                chunk = r.read(half*2)
                assert len(chunk)==half*2
                f.write(chunk[:half])
        print(dest,dest.stat().st_size,flush=True)
