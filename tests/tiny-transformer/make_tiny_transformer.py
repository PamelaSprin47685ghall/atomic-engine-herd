#!/usr/bin/env python3
"""Write a tiny, real, loadable llama-architecture transformer GGUF.

Why this exists: the reader-attention seam hooks llm_graph_context::build_attn,
which only runs for models that have attention layers. The only GGUF on this
machine was a mamba checkpoint, whose graph has no attention at all, so the seam
reported "observed 0 calls" -- and equality of nll between lane-on and lane-off
proved nothing about the hook.

This writes a genuine LLM_ARCH_LLAMA checkpoint: three blocks, F32 weights, a
small vocabulary with the 256 byte-fallback tokens a llama tokenizer needs. It is
small enough to load in a second and large enough that the attention path is
exercised. The weights are deterministic (fixed seed), so two runs are comparable.

The GGUF container is written directly, with no third-party modules, following the
same layout as the repo's mamba generator: metadata, tensor infos, then data at the
aligned position after the infos. (The alignment detail matters: a loader reads
tensor data from that position, not from the summed tensor sizes.)

usage: python3 tests/tiny-transformer/make_tiny_transformer.py out.gguf
"""

import random
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

T_UINT32 = 4
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9

GGML_TYPE_F32 = 0

TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_BYTE = 6

ALIGNMENT = 32


def align(n, a=ALIGNMENT):
    return (n + a - 1) // a * a


class Writer:
    def __init__(self):
        self.kv = []

    def add(self, key, value):
        self.kv.append((key, value))

    def _string(self, s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _value(self, v):
        if isinstance(v, bool):
            return struct.pack("<I", T_BOOL) + struct.pack("<B", 1 if v else 0)
        if isinstance(v, int):
            return struct.pack("<I", T_UINT32) + struct.pack("<I", v)
        if isinstance(v, float):
            return struct.pack("<I", T_FLOAT32) + struct.pack("<f", v)
        if isinstance(v, str):
            return struct.pack("<I", T_STRING) + self._string(v)
        if isinstance(v, list):
            if not v:
                raise ValueError("empty arrays are not used here")
            if isinstance(v[0], str):
                body = b"".join(self._string(x) for x in v)
                return struct.pack("<I", T_ARRAY) + struct.pack("<I", T_STRING) + struct.pack("<Q", len(v)) + body
            if isinstance(v[0], float):
                body = b"".join(struct.pack("<f", x) for x in v)
                return struct.pack("<I", T_ARRAY) + struct.pack("<I", T_FLOAT32) + struct.pack("<Q", len(v)) + body
            if isinstance(v[0], int):
                body = b"".join(struct.pack("<i", x) for x in v)
                return struct.pack("<I", T_ARRAY) + struct.pack("<I", 5) + struct.pack("<Q", len(v)) + body
        raise TypeError(f"unsupported metadata value: {type(v)}")

    def write(self, path, tensors):
        offsets = []
        off = 0
        for _, _, data in tensors:
            off = align(off)
            offsets.append(off)
            off += len(data)

        with open(path, "wb") as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(tensors)))
            f.write(struct.pack("<Q", len(self.kv)))
            for key, value in self.kv:
                f.write(self._string(key))
                f.write(self._value(value))
            for (name, dims, _), tensor_off in zip(tensors, offsets):
                f.write(self._string(name))
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", GGML_TYPE_F32))
                f.write(struct.pack("<Q", tensor_off))
            # Tensor data starts at the aligned position after the infos.
            data_start = align(f.tell())
            f.write(b"\x00" * (data_start - f.tell()))
            for (_, _, data), tensor_off in zip(tensors, offsets):
                f.write(b"\x00" * (tensor_off - (f.tell() - data_start)))
                f.write(data)
            f.write(b"\x00" * (align(f.tell()) - f.tell()))


def f32_tensor(rng, dims, scale=0.1, value=None):
    n = 1
    for d in dims:
        n *= d
    if value is None:
        vals = [rng.gauss(0.0, scale) for _ in range(n)]
    else:
        vals = [value] * n
    return struct.pack("<%df" % n, *vals)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]

    n_embd  = 32
    n_layer = 3
    n_head  = 4
    n_head_kv = 4          # no GQA: keeps the K/V layout trivially rectangular
    n_ff    = 64
    n_ctx   = 128
    n_rot   = n_embd // n_head  # 8: full rotary over each head

    rng = random.Random(20261124)

    w = Writer()
    w.add("general.architecture", "llama")
    w.add("general.name", "tiny-transformer")
    w.add("llama.context_length", n_ctx)
    w.add("llama.embedding_length", n_embd)
    w.add("llama.block_count", n_layer)
    w.add("llama.feed_forward_length", n_ff)
    w.add("llama.attention.head_count", n_head)
    w.add("llama.attention.head_count_kv", n_head_kv)
    w.add("llama.attention.layer_norm_rms_epsilon", 1e-5)
    w.add("llama.rope.dimension_count", n_rot)
    w.add("llama.rope.freq_base", 10000.0)

    # A llama/SPM vocabulary needs the 256 byte-fallback tokens so the tokenizer
    # accepts arbitrary text; this test feeds token ids directly, but the file
    # stays usable by stock tools.
    tokens = ["<unk>", "<s>", "</s>"]
    token_types = [TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL]
    tokens += ["<0x%02X>" % i for i in range(256)]
    token_types += [TOKEN_TYPE_BYTE] * 256
    words = ["a", "b", "c", "d"]
    tokens += words
    token_types += [TOKEN_TYPE_NORMAL] * len(words)
    n_vocab = len(tokens)
    w.add("tokenizer.ggml.model", "llama")
    w.add("tokenizer.ggml.tokens", tokens)
    w.add("tokenizer.ggml.scores", [0.0] * len(tokens))
    w.add("tokenizer.ggml.token_type", token_types)

    tensors = []

    def add(name, dims, scale=0.1, value=None):
        dims = list(dims)
        tensors.append((name, dims, f32_tensor(rng, dims, scale, value)))

    add("token_embd.weight", (n_embd, n_vocab), scale=1.0)
    add("output_norm.weight", (n_embd,), value=1.0)
    add("output.weight", (n_embd, n_vocab), scale=1.0)

    for il in range(n_layer):
        p = f"blk.{il}."
        add(p + "attn_norm.weight", (n_embd,), value=1.0)
        add(p + "attn_q.weight", (n_embd, n_head * n_rot), scale=0.3)
        add(p + "attn_k.weight", (n_embd, n_head_kv * n_rot), scale=0.3)
        add(p + "attn_v.weight", (n_embd, n_head_kv * n_rot), scale=0.3)
        add(p + "attn_output.weight", (n_head * n_rot, n_embd), scale=0.3)
        add(p + "ffn_norm.weight", (n_embd,), value=1.0)
        add(p + "ffn_gate.weight", (n_embd, n_ff), scale=0.2)
        add(p + "ffn_up.weight", (n_embd, n_ff), scale=0.2)
        add(p + "ffn_down.weight", (n_ff, n_embd), scale=0.2)

    import os

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    w.write(out, tensors)
    print(f"wrote {out}: arch=llama layers={n_layer} n_embd={n_embd} n_head={n_head} "
          f"n_rot={n_rot} n_ff={n_ff} vocab={n_vocab} tensors={len(tensors)}")
    print(f"token ids: a={tokens.index('a')} b={tokens.index('b')} c={tokens.index('c')}")


if __name__ == "__main__":
    main()
