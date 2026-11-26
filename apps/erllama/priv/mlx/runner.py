"""Private, length-prefixed erllama MLX worker. One process owns one sequence."""
import base64
import json
import os
from pathlib import Path
import struct
import sys

VERSION = 1
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


class QwenSequence:
    def __init__(self, path, limit):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step, generation_stream
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
        self.generate_step = generate_step
        self.stream = generation_stream
        self.limit = min(limit, config["text_config"]["max_position_embeddings"])
        self.byte_decoder = _byte_decoder()
        self.added = {t["id"]: t for t in tokenizer["added_tokens"]}
        self.reset()

    def reset(self):
        self.cache = self.make_cache(self.model)
        self.position = 0
        self.last_prompt_token = None
        self.generator = None

    def prefill(self, tokens):
        # MLX-LM feeds the last prompt token through its single-token decode path.
        if self.last_prompt_token is not None:
            tokens = [self.last_prompt_token, *tokens]
        if self.generator is not None:
            self.generator.close()
            self.generator = None
        self.last_prompt_token = tokens[-1]
        with self.mx.stream(self.stream):
            for start in range(0, len(tokens) - 1, 512):
                chunk = tokens[start:min(start + 512, len(tokens) - 1)]
                self.model(self.mx.array([chunk]), cache=self.cache)
                self.mx.eval([c.state for c in self.cache])

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
        if op == "clear":
            self.reset()
            return None
        if op == "encode":
            return self.tokenizer.encode(arg)
        if op == "pieces":
            return self.pieces(*arg)
        if op == "prefill":
            if not arg or min(arg) < 0 or max(arg) >= len(self.tokenizer):
                raise ValueError("invalid prompt tokens")
            if self.position + len(arg) > self.limit:
                raise OverflowError()
            self.prefill(arg)
            self.position += len(arg)
            return None
        if op == "decode" and self.position:
            if self.position >= self.limit:
                raise OverflowError()
            if self.generator is None:
                self.generator = self.generate_step(
                    self.mx.array([self.last_prompt_token]), self.model,
                    prompt_cache=self.cache, max_tokens=-1,
                )
                self.last_prompt_token = None
            token, _ = next(self.generator)
            self.position += 1
            return [token, int(token in self.tokenizer.eos_token_ids)]
        raise ValueError("invalid operation or sequence state")


def main():
    with os.fdopen(os.dup(1), "wb") as out:
        os.dup2(2, 1)  # Native libraries can also print on stdout.
        try:
            seq = QwenSequence(Path(sys.argv[1]), int(sys.argv[2]))
        except Exception as exc:
            send(out, "load_error", str(exc))
            return
        send(out, "ok", "ready")
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
