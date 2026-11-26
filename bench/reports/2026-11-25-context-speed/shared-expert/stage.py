"""Stage layer zero for an even two-rank split. Run in an empty scratch directory."""

import json
import math
import struct
import urllib.request
from pathlib import Path

import gguf
import numpy as np

HF = "https://huggingface.co/"
REV = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
BASE = HF + "deepseek-ai/DeepSeek-V4-Flash-0731/resolve/" + REV + "/"
DS4 = (
    HF + "antirez/deepseek-v4-gguf/resolve/f71f23d552d664e523b422157b2befbf74040380/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
)


def byte_range(url, first, count):
    end = first + count - 1
    req = urllib.request.Request(
        f"{url}?slice={first}-{end}", headers={"Range": f"bytes={first}-{end}"}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        rows, size = resp.headers["Content-Range"].split("/")
        if resp.status != 206 or rows != f"bytes {first}-{end}":
            raise ValueError("unexpected byte range")
        blob = resp.read(count + 1)
        if len(blob) != count:
            raise IOError("truncated tensor range")
        return blob, int(size)


def source_tensor(url, offset, meta):
    lo, hi = meta["data_offsets"]
    return np.frombuffer(byte_range(url, offset + lo, hi - lo)[0], np.uint8).reshape(
        meta["shape"]
    )


def dequantize(shard, start, w, s):
    assert w["dtype"] == "F8_E4M3" and s["dtype"] == "F8_E8M0"
    raw = source_tensor(shard, start, w)
    scale = source_tensor(shard, start, s)
    assert not np.any(scale == 255) and not np.any((raw & 127) == 127)
    expo = ((raw >> 3) & 15).astype(np.int16)
    frac = (raw & 7).astype(np.float32)
    values = np.where(expo == 0, np.ldexp(frac, -9), np.ldexp(1 + frac / 8, expo - 7))
    values = np.where(raw >= 128, -values, values)
    rows, cols = raw.shape
    assert tuple(s["shape"]) == (rows // 128, cols // 128)
    weights = np.ldexp(
        values.reshape(rows // 128, 128, cols // 128, 128),
        scale.astype(np.int16)[:, None, :, None] - 127,
    ).reshape(rows, cols)
    return weights


def main():
    shard = BASE + "model-00002-of-00048.safetensors"
    head = struct.unpack("<Q", byte_range(shard, 0, 8)[0])[0]
    tensors = json.loads(byte_range(shard, 8, head)[0])
    header, size = byte_range(DS4, 0, 8 << 20)
    # Map only the GGUF directory. Tensor payloads are fetched separately.
    with open("directory.gguf", "xb") as f:
        f.write(header)
        f.truncate(size)
    model = gguf.GGUFReader("directory.gguf")
    directory = {t.name: t for t in model.tensors}
    clamp_field = model.fields.get(
        "deepseek4.swiglu_clamp_shexp", model.fields["deepseek4.swiglu_clamp_exp"]
    )
    limit = float(np.asarray(clamp_field.contents()).flatten()[0])
    Path("clamp.txt").write_text(str(limit))
    for role, name in [("gate", "w1"), ("up", "w3"), ("down", "w2")]:
        prefix = "layers.0.ffn.shared_experts." + name
        w, s = tensors[prefix + ".weight"], tensors[prefix + ".scale"]
        weights = dequantize(shard, head + 8, w, s)
        rows, cols = weights.shape
        tensor = directory[f"blk.0.ffn_{role}_shexp.weight"]
        assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0 and list(
            tensor.shape
        ) == [cols, rows]
        packed = byte_range(DS4, int(tensor.data_offset), int(tensor.n_bytes))[0]
        blocks = np.frombuffer(packed, np.uint8).reshape(rows, cols // 32, 34)
        if role == "down":
            weights, blocks = weights[:, : cols // 2], blocks[:, : cols // 64]
        else:
            weights, blocks = weights[: rows // 2], blocks[: rows // 2]
        restored = gguf.quants.dequantize(
            np.ascontiguousarray(blocks).reshape(-1, 34), gguf.GGMLQuantizationType.Q8_0
        ).reshape(weights.shape)
        err = math.sqrt(np.mean((weights - restored) ** 2) / np.mean(weights**2))
        print(role, weights.shape, "source vs ds4 Q8 relative RMS", err, flush=True)
        assert np.isfinite(err) and err < 0.025
        np.ascontiguousarray(weights, dtype="<f4").tofile(role + ".f32")
        np.ascontiguousarray(blocks).tofile(role + ".q8")
    print("shared SwiGLU limit", limit)


if __name__ == "__main__":
    main()
