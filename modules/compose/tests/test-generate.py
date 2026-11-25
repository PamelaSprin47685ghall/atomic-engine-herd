#!/usr/bin/env python3
"""End-to-end regression test for llama-compose.

Why a model is required, and why the test is gated rather than skipped:

The defect this guards against -- the final prompt chunk not requesting logits --
only shows up when a real context runs: the sampler receives no logits, upstream
logs `invalid logits id -1 ... n_outputs=0`, and the driver fails at the first
token. Nothing below `main()` can reproduce it, because the cause is a decision
made in the decode plan. A test that "skips" without a model would look green
forever without ever exercising the path, which is the fake-green shape this
project refuses; so the test is registered only when a model is provided, and
its absence is visible in the CTest list rather than in a passing result.

What each check pins:

  * exit 0 with a non-empty token sequence -- the whole decode/sample path;
  * the specific logits-error text is absent -- names the failure it prevents;
  * two runs produce identical tokens -- the migration did not introduce hidden
    state between runs (the JSON's timing field is excluded, since it varies);
  * a prompt that fits one batch and one that is a single token both work -- the
    last-chunk rule must hold at the degenerate edges, where "last" and "only"
    coincide;
  * `-n 0` exits 0 -- an argument contract, not a behaviour claim;
  * an unknown option is refused -- the CLI does not silently accept what it
    does not understand.
"""

import argparse
import json
import subprocess
import sys


def run(binary, args):
    completed = subprocess.run([binary, *args], capture_output=True, text=True)
    return completed.returncode, completed.stdout, completed.stderr


def tokens_of(binary, model, prompt, n):
    rc, out, err = run(binary, ["-m", model, "-p", prompt, "-n", str(n), "-c", "128", "--json"])
    if rc != 0:
        return None, f"exit {rc}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
    try:
        return json.loads(out.strip().splitlines()[-1]), None
    except (ValueError, IndexError) as error:
        return None, f"unparsable JSON ({error})\n{out}"


def gguf_architecture(path):
    """Read `general.architecture` out of a GGUF header, or None.

    This exists because the test used to be pointed at a file named `tiny-mamba.gguf` that was
    byte-identical to a llama model, and nothing in the assertions noticed. A name is not a
    claim; the header is. Unknown value types end the scan rather than guessing a width -- a
    wrong skip would read the next key's bytes as this key's value.
    """
    import struct

    with open(path, "rb") as handle:
        if handle.read(4) != b"GGUF":
            return None
        version, = struct.unpack("<I", handle.read(4))
        if version < 2:
            return None
        handle.read(8)  # n_tensors
        n_kv, = struct.unpack("<Q", handle.read(8))
        for _ in range(n_kv):
            key_len, = struct.unpack("<Q", handle.read(8))
            key = handle.read(key_len).decode("utf-8", "replace")
            value_type, = struct.unpack("<I", handle.read(4))
            value = None
            if value_type == 8:
                length, = struct.unpack("<Q", handle.read(8))
                value = handle.read(length).decode("utf-8", "replace")
            elif value_type in (0, 1, 7):
                value = struct.unpack("<B", handle.read(1))[0]
            elif value_type in (2, 3):
                value = struct.unpack("<H", handle.read(2))[0]
            elif value_type in (4, 5):
                value = struct.unpack("<I", handle.read(4))[0]
            elif value_type in (6, 10, 11, 12):
                handle.read(8 if value_type == 12 else 4)
            else:
                # Arrays and anything unknown: stop. `general.architecture` is a scalar string
                # and upstream writes it first; anything else means this is not the layout
                # assumed here, and a guess would be worse than an honest "unread".
                return None
            if key == "general.architecture":
                return value
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--expect-arch",
        default=None,
        help="Assert the model's GGUF general.architecture equals this value.",
    )
    args = parser.parse_args()

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    if args.expect_arch is not None:
        arch = gguf_architecture(args.model)
        check(
            arch == args.expect_arch,
            f"model architecture is {arch!r}, expected {args.expect_arch!r} "
            f"({args.model}: the file header, not the file name, is the claim)",
        )

    # 1. The path that broke: prompt decode, then sampling from the last row.
    payload, error = tokens_of(args.binary, args.model, "hello world", 6)
    if error:
        failures.append(f"generation failed: {error}")
    else:
        check(bool(payload.get("tokens")), "generation produced no tokens")
        check(len(payload["tokens"]) <= 6, "generation exceeded the requested length")
        check(payload.get("prompt_tokens", 0) > 0, "prompt was not tokenized")

    # 2. The same shape as (1) but with the message that the missing logits
    #    request produced, so a regression names itself instead of looking like a
    #    sampler bug.
    rc, out, err = run(args.binary, ["-m", args.model, "-p", "hello world", "-n", "2", "-c", "64", "--json"])
    check("did not request output_logits" not in (out + err),
          "the driver reported a missing logits request: the decode plan regressed")

    # 3. Determinism of the sampled tokens (the timing field is excluded).
    first, error_a = tokens_of(args.binary, args.model, "hello world", 6)
    second, error_b = tokens_of(args.binary, args.model, "hello world", 6)
    if error_a or error_b:
        failures.append(f"determinism run failed: {error_a or error_b}")
    else:
        check(first["tokens"] == second["tokens"],
              f"two runs disagreed: {first['tokens']} vs {second['tokens']}")

    # 4. Degenerate edges: the only chunk is also the last chunk.
    single, error = tokens_of(args.binary, args.model, "hi", 3)
    if error:
        failures.append(f"single-token prompt failed: {error}")
    else:
        check(len(single["tokens"]) <= 3, "single-token prompt generated too much")

    # 5. Argument contracts.
    rc, _, _ = run(args.binary, ["-m", args.model, "-p", "hi", "-n", "0", "-c", "64", "--json"])
    check(rc == 0, f"-n 0 must exit 0, got {rc}")

    rc, _, err = run(args.binary, ["-m", args.model, "-p", "hi", "--not-a-flag"])
    check(rc != 0, "an unknown option must be refused")

    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        print(f"\n{len(failures)} generation contract violation(s)", file=sys.stderr)
        return 1
    print("compose generation contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
