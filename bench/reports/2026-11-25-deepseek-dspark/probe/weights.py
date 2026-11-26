"""Add any missing shared projections to a scratch draft head, without requantizing."""
import os
import struct
import sys
from pathlib import Path


def index(path):
    with path.open('rb') as f:
        def number(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        def string():
            return f.read(number('Q')).decode()

        widths = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8]

        def skip(kind, count=1):
            if kind == 9:
                for _ in range(count):
                    skip(number('I'), number('Q'))
            elif kind == 8:
                for _ in range(count):
                    f.seek(number('Q'), 1)
            else:
                f.seek(widths[kind] * count, 1)

        assert f.read(4) == b'GGUF' and number('I') == 3
        n, nk = number('Q'), number('Q')
        align = 32
        for _ in range(nk):
            key, kind = string(), number('I')
            if key == 'general.alignment':
                assert kind == 4
                align = number('I')
            else:
                skip(kind)
        metadata_end = f.tell()
        entries = {}
        for _ in range(n):
            start = f.tell()
            name = string()
            dims = [number('Q') for _ in range(number('I'))]
            kind, off = number('I'), number('Q')
            entries[name] = (start, f.tell(), off, dims, kind)
        end = f.tell()
        return metadata_end, end, (end + align - 1) // align * align, align, entries


def span(src, dst, start, length):
    src.seek(start)
    while length:
        block = src.read(min(length, 16 << 20))
        if not block:
            raise EOFError()
        dst.write(block)
        length -= len(block)


def build(head, target, output):
    _, he, hd, align, ht = index(head)
    _, _, td, _, tt = index(target)
    wanted = [n for n in ('token_embd.weight', 'output.weight', 'output.scale') if n in tt and n not in ht]
    for n in ('token_embd.weight', 'output.weight'):
        assert n in ht or n in tt
    print('copied tensors', wanted, flush=True)
    if not wanted:
        output.symlink_to(head.resolve())
        return
    with head.open('rb') as h, target.open('rb') as t, output.open('xb') as out:
        span(h, out, 0, 8)
        out.write(struct.pack('<Q', len(ht) + len(wanted)))
        span(h, out, 16, he - 16)
        offsets = sorted(e[2] for e in tt.values()) + [target.stat().st_size - td]
        end = head.stat().st_size - hd
        pieces = []
        for name in wanted:
            start, stop, off, dims, kind = tt[name]
            print(name, dims, kind, flush=True)
            length = offsets[offsets.index(off)+1] - off
            end = (end + align-1)//align*align
            span(t, out, start, stop-start-8)
            out.write(struct.pack('<Q', end))
            pieces.append((end, td+off, length))
            end += length
        data = (out.tell() + align-1)//align*align
        out.seek(data)
        span(h, out, hd, head.stat().st_size-hd)
        for off, start, length in pieces:
            out.seek(data+off)
            span(t, out, start, length)
    assert len(index(output)[4]) == len(ht) + len(wanted)


if __name__ == '__main__':
    build(*map(Path, sys.argv[1:]))
