"""Build the GX10 candidate in scratch with a verified baseline and native quantize executable."""

from contextlib import nullcontext
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import urllib.request

import gguf
import numpy as np

from stage import BASE, REV, byte_range, dequantize

BASE_SHA = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
Q4_SHA = "68bc065075b44dab261a80cb0dd81c6629af085e591f283f41e0df7ca4316d0f"
ROLES = {"gate": "w1", "down": "w2", "up": "w3"}
TARGET = re.compile(r"blk\.(\d+)\.ffn_(gate|down|up)_shexp\.weight")
CHUNK = 16 << 20


@functools.cache
def shard_header(shard):
    size = struct.unpack("<Q", byte_range(BASE + shard, 0, 8)[0])[0]
    return size + 8, json.loads(byte_range(BASE + shard, 8, size)[0])


def load_weights(index, prefix):
    weight = prefix + ".weight"
    scale = prefix + ".scale"
    shard = index[weight]
    if index[scale] != shard:
        raise ValueError("unexpected cross-shard scaling: " + prefix)
    start, meta = shard_header(shard)
    return dequantize(BASE + shard, start, meta[weight], meta[scale])


def tensor_path(name, fmt):
    return Path(name + (".q5" if fmt == "Q5_K" else ".q4"))


def quantize_all(fmt):
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=90) as f:
        index = json.load(f)["weight_map"]
    for layer in range(43):
        for role, w in ROLES.items():
            name = f"blk.{layer}.ffn_{role}_shexp.weight"
            weights = load_weights(index, f"layers.{layer}.ffn.shared_experts.{w}")
            rows, cols = weights.shape
            assert (rows, cols) == ((4096, 2048) if role == "down" else (2048, 4096))
            assert np.all(np.isfinite(weights))
            with open(name + ".part", "wb") as f:
                subprocess.run(
                    ["./quantize", str(cols), str(rows), fmt],
                    input=weights.astype("<f4", copy=False).tobytes(),
                    stdout=f,
                    check=True,
                )
            part = Path(name + ".part")
            assert (
                part.stat().st_size
                == rows
                * cols
                // 256
                * gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[fmt]][1]
            )
            part.replace(tensor_path(name, fmt))
        print("quantized layer", layer, flush=True)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def assemble(fmt, baseline):
    target = gguf.GGMLQuantizationType[fmt]
    output = Path(
        f"DeepSeek-V4-Flash-0731-erllama-GX10-Shared{fmt.replace('_', '')}.gguf"
    )
    if output.exists():
        raise FileExistsError(output)
    template_sha = digest(baseline)
    if template_sha == BASE_SHA:
        shared_type = gguf.GGMLQuantizationType.Q8_0
    elif fmt == "Q5_K" and template_sha == Q4_SHA:
        shared_type = gguf.GGMLQuantizationType.Q4_K
    else:
        raise ValueError("unexpected template checksum")
    src = gguf.GGUFReader(baseline)
    with baseline.open("rb") as f:
        head = bytearray(f.read(src.data_offset))
    layout = []
    changed = []
    offset = 0
    for t in src.tensors:
        ty = t.tensor_type
        replace = TARGET.fullmatch(t.name)
        nbytes = int(t.n_bytes)
        if replace:
            assert ty == shared_type
            ty = target
            nbytes = int(t.n_elements) // 256 * gguf.GGML_QUANT_SIZES[target][1]
            assert tensor_path(t.name, fmt).stat().st_size == nbytes
            changed.append(t.name)
        # The last fields of every tensor directory entry are its type and offset.
        field = t.field
        pos = int(field.offset) + sum(p.nbytes for p in field.parts[:-2])
        old = struct.unpack_from("<IQ", head, pos)
        assert old == (int(t.tensor_type), int(t.data_offset) - src.data_offset)
        struct.pack_into("<IQ", head, pos, int(ty), offset)
        layout.append((t, replace, nbytes, offset))
        offset = (offset + nbytes + src.alignment - 1) // src.alignment * src.alignment
    assert len(changed) == 129
    assert set(changed) == {
        f"blk.{i}.ffn_{r}_shexp.weight" for i in range(43) for r in ROLES
    }
    part = output.with_suffix(".part")
    space = os.statvfs(".")
    if space.f_bavail * space.f_frsize < len(head) + offset + (4 << 30):
        raise OSError("insufficient disk space")
    fingerprints = {}
    with baseline.open("rb") as base, part.open("xb") as out:
        out.write(head)
        for t, replace, size, where in layout:
            with (
                tensor_path(t.name, fmt).open("rb")
                if replace
                else nullcontext(base) as inp
            ):
                if not replace:
                    inp.seek(t.data_offset)
                out.seek(len(head) + where)
                left = size
                sha = hashlib.sha256()
                while left:
                    data = inp.read(min(CHUNK, left))
                    if not data:
                        raise EOFError(t.name)
                    out.write(data)
                    sha.update(data)
                    left -= len(data)
                fingerprints[t.name] = sha.hexdigest()
        out.truncate(len(head) + offset)
    dest = gguf.GGUFReader(part)
    assert list(src.fields) == list(dest.fields) and len(src.tensors) == len(
        dest.tensors
    )
    for key in src.fields:
        assert all(
            np.array_equal(a, b)
            for a, b in zip(src.fields[key].parts, dest.fields[key].parts, strict=True)
        )
    with part.open("rb") as f:
        for (original, replace, size, _), t in zip(layout, dest.tensors, strict=True):
            assert (
                original.name == t.name
                and list(original.shape) == list(t.shape)
                and t.n_bytes == size
            )
            assert t.tensor_type == (target if replace else original.tensor_type)
            f.seek(t.data_offset)
            sha = hashlib.sha256()
            left = size
            while left:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    raise EOFError(t.name)
                sha.update(chunk)
                left -= len(chunk)
            assert sha.hexdigest() == fingerprints[t.name], t.name
    checksum = digest(part)
    result = dict(
        file=output.name,
        bytes=part.stat().st_size,
        sha256=checksum,
        base_sha256=BASE_SHA,
        template_sha256=template_sha,
        source_revision=REV,
        tensors=len(layout),
        changed=changed,
    )
    part.rename(output)
    Path("manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print("verified", output, checksum, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["quantize", "assemble"])
    p.add_argument("--format", choices=["Q4_K", "Q5_K"], default="Q4_K")
    p.add_argument("--template", type=Path, default=Path("baseline.gguf"))
    args = p.parse_args()
    if args.action == "quantize":
        quantize_all(args.format)
    else:
        assemble(args.format, args.template)
