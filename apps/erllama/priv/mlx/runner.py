"""Private, length-prefixed erllama MLX worker. One process owns a model and its sequence slots."""
import base64
import hashlib
import json
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
import os
from pathlib import Path
import struct
import sys

VERSION = 2
MAX_FRAME = 64 * 1024 * 1024


def send(out, status, value):
    payload = json.dumps([VERSION, status, value], separators=(",", ":")).encode()
    if len(payload) > MAX_FRAME:
        raise ValueError("reply exceeds frame limit")
    out.write(struct.pack(">I", len(payload)) + payload)
    out.flush()


def receive():
    header = sys.stdin.buffer.read(4)
    if not header:
        return None
    size, = struct.unpack(">I", header)
    if not 0 < size <= MAX_FRAME:
        raise ValueError("invalid frame size")
    payload = sys.stdin.buffer.read(size)
    if len(payload) != size:
        raise EOFError("incomplete frame")
    version, command, argument = json.loads(payload)
    if version != VERSION:
        raise ValueError("unsupported protocol")
    return command, argument


@dataclass
class Sequence:
    layers: list
    tokens: list
    pending: bool = False


class QwenModel:
    def __init__(self, path, limit, slots, scratch):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generation_stream
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.tokenizer_utils import _byte_decoder

        config = json.loads((path / "config.json").read_text())
        tokenizer = json.loads((path / "tokenizer.json").read_text())
        if config.get("model_type") != "qwen3_5" or tokenizer["decoder"]["type"] != "ByteLevel":
            raise ValueError("requires a Qwen3.5-family text model with a byte-level tokenizer")
        if not mx.metal.is_available():
            raise RuntimeError("Metal device unavailable")
        mx.set_default_device(mx.gpu)
        self.mx = mx
        self.model, self.tokenizer = load(path, tokenizer_config={"trust_remote_code": False})
        self.make_cache = make_prompt_cache
        self.stream = generation_stream
        self.limit = min(limit, config["text_config"]["max_position_embeddings"])
        self.byte_decoder = _byte_decoder()
        self.added = {t["id"]: t for t in tokenizer["added_tokens"]}
        self.slots = [Sequence(make_prompt_cache(self.model), []) for _ in range(slots)]
        self.batch_slots = []
        self.batch_layers = None
        self.file = str(Path(scratch) / "state.safetensors")
        digest = hashlib.sha256()
        for file in sorted(path.iterdir()):
            if file.suffix in (".json", ".safetensors", ".jinja", ".model", ".txt"):
                digest.update(file.name.encode() + b"\0")
                with file.open("rb") as stream:
                    digest.update(hashlib.file_digest(stream, "sha256").digest())
        self.fingerprint = digest.hexdigest()
        self.format = f"mlx-qwen-kv-v1/{version('mlx')}/{version('mlx-lm')}/{self.limit}"

    def prefill(self, seq, tokens):
        # Keep one pending token so an exact cache hit can re-prime hybrid layers.
        inputs = seq.tokens[-1:] + tokens if seq.pending else tokens
        for start in range(0, len(inputs) - 1, 512):
            chunk = inputs[start:min(start + 512, len(inputs) - 1)]
            self.model(self.mx.array([chunk]), cache=seq.layers)
            self.mx.eval([layer.state for layer in seq.layers])
        seq.tokens.extend(tokens)
        seq.pending = True

    def split_batch(self):
        if self.batch_slots:
            for i, slot in enumerate(self.batch_slots):
                self.slot(slot).layers = [layer.extract(i) for layer in self.batch_layers]
            self.batch_slots = []
            self.batch_layers = None

    def decode_layers(self, slots):
        if len(slots) == 1:
            return self.slot(slots[0]).layers
        if not self.batch_slots:
            rows = [self.slot(slot) for slot in slots]
            self.batch_layers = [group[0].merge(group) for group in zip(*(row.layers for row in rows))]
            self.batch_slots = slots
            for row in rows:
                row.layers = None
        return self.batch_layers

    def step(self, ops):
        if len({slot for slot, _, _ in ops}) != len(ops):
            raise ValueError("duplicate slot")
        decode = []
        results = {}
        for slot, op, ids in ops:
            seq = self.slot(slot)
            if op == "prefill" and ids and min(ids) >= 0 and max(ids) < len(self.tokenizer):
                count = len(ids)
            elif op == "decode" and seq.pending:
                count = 1
                decode.append(slot)
            else:
                raise ValueError("invalid step")
            if len(seq.tokens) + count > self.limit:
                raise OverflowError()
        with self.mx.stream(self.stream):
            if decode != self.batch_slots:
                self.split_batch()
            for slot, op, ids in ops:
                if op == "prefill":
                    self.prefill(self.slot(slot), ids)
                    results[slot] = None
            if decode:
                rows = [self.slot(slot) for slot in decode]
                layers = self.decode_layers(decode)
                inputs = self.mx.array([[row.tokens[-1]] for row in rows])
                logits = self.model(inputs, cache=layers)[:, -1, :]
                ids = self.mx.argmax(logits, axis=-1).tolist()
                for slot, row, token in zip(decode, rows, ids):
                    row.tokens.append(token)
                    results[slot] = [token, int(token in self.tokenizer.eos_token_ids)]
        return [[slot, results[slot]] for slot, _, _ in ops]

    def slot(self, slot):
        if not isinstance(slot, int) or not 0 <= slot < len(self.slots):
            raise ValueError("invalid slot")
        return self.slots[slot]

    def save(self, slot, tokens):
        seq = self.slot(slot)
        if not seq.pending or tokens != seq.tokens:
            raise ValueError("checkpoint token boundary mismatch")
        tensors = {"tokens": self.mx.array(tokens, dtype=self.mx.int32)}
        for i, layer in enumerate(seq.layers):
            if layer.empty():
                continue
            if type(layer).__name__ == "KVCache":
                values = [value[..., :layer.offset, :] for value in (layer.keys, layer.values)]
            else:
                values = layer.cache
            for j, value in enumerate(values):
                tensors[f"{i}.{j}"] = value
        self.mx.save_safetensors(self.file, tensors, {"model": self.fingerprint, "format": self.format})

    def restore(self, slot):
        self.slot(slot)
        tensors, meta = self.mx.load(self.file, return_metadata=True)
        if meta != {"model": self.fingerprint, "format": self.format}:
            raise ValueError("incompatible checkpoint")
        ids = tensors.pop("tokens")
        if ids.dtype != self.mx.int32 or ids.ndim != 1 or not 0 < ids.size <= self.limit:
            raise ValueError("invalid checkpoint token array")
        tokens = ids.tolist()
        if min(tokens) < 0 or max(tokens) >= len(self.tokenizer):
            raise ValueError("invalid checkpoint tokens")
        layers = self.make_cache(self.model)
        if len(tokens) > 1:
            for i, layer in enumerate(layers):
                k, v = (tensors.pop(f"{i}.{j}") for j in range(2))
                if not all(self.mx.issubdtype(t.dtype, self.mx.floating) for t in (k, v)):
                    raise ValueError("invalid checkpoint dtype")
                weights = self.model.layers[i]
                if weights.is_linear:
                    linear = weights.linear_attn
                    if (k.shape != (1, linear.conv_kernel_size - 1, linear.conv_dim)
                            or v.shape != (1, linear.num_v_heads, linear.head_v_dim, linear.head_k_dim)):
                        raise ValueError("invalid recurrent checkpoint shape")
                    layer.cache = [k, v]
                else:
                    attn = weights.self_attn
                    shape = (1, attn.num_key_value_heads, len(tokens) - 1, attn.head_dim)
                    if k.shape != shape or v.shape != shape:
                        raise ValueError("invalid attention checkpoint shape")
                    layer.state = k, v, len(tokens) - 1
        if tensors:
            raise ValueError("unexpected checkpoint tensors")
        self.mx.eval([layer.state for layer in layers])
        self.slots[slot] = Sequence(layers, tokens, True)

    def pieces(self, tokens, special):
        # A token can end inside a UTF-8 character. Preserve its bytes across chunks.
        data = bytearray()
        for token_id in tokens:
            added = self.added.get(token_id)
            if added:
                if special or not added["special"]:
                    data.extend(added["content"].encode())
            else:
                piece = self.tokenizer.convert_ids_to_tokens(token_id)
                if piece is None:
                    raise ValueError("invalid token id")
                data.extend(self.byte_decoder[ch] for ch in piece)
        return base64.b64encode(data).decode()

    def handle(self, op, arg):
        if op in ("clear", "trim_last", "pack", "unpack") and self.batch_slots:
            with self.mx.stream(self.stream):
                self.split_batch()
        if op == "clear":
            self.slot(arg)
            self.slots[arg] = Sequence(self.make_cache(self.model), [])
            return None
        if op == "trim_last":
            slot, length = arg
            seq = self.slot(slot)
            if not seq.pending or length != len(seq.tokens):
                raise ValueError("cannot trim recurrent state")
            seq.tokens.pop()
            seq.pending = False
            return None
        if op == "pack":
            try:
                return self.save(*arg)
            except (RuntimeError, OSError) as exc:
                raise ValueError("checkpoint write failed") from exc
        if op == "unpack":
            try:
                return self.restore(arg)
            except (ValueError, KeyError, IndexError, TypeError, RuntimeError, OSError) as exc:
                raise ValueError("invalid checkpoint") from exc
        if op == "encode":
            return self.tokenizer.encode(arg)
        if op == "pieces":
            return self.pieces(*arg)
        if op == "step":
            return self.step(arg)
        raise ValueError("invalid operation or sequence state")


def main():
    with os.fdopen(os.dup(1), "wb") as out, tempfile.TemporaryDirectory(prefix="erllama-mlx-") as scratch:
        os.dup2(2, 1)  # Native libraries can also print on stdout.
        try:
            seq = QwenModel(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), scratch)
        except Exception as exc:
            send(out, "load_error", str(exc))
            return
        send(out, "ok", [seq.file, seq.fingerprint, seq.format, seq.tokenizer.chat_template,
                         seq.tokenizer.bos_token or "", seq.tokenizer.eos_token or ""])
        while (request := receive()) is not None:
            try:
                result = seq.handle(*request)
            except OverflowError:
                send(out, "error", "context_full")
            except ValueError as exc:
                send(out, "error", str(exc))
            else:
                send(out, "ok", result)


if __name__ == "__main__":
    main()
