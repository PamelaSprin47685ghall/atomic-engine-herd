#!/usr/bin/env python3
"""The refusal contract of tq-rerot-compose.

RERoT is defined by the fork's reader subsystem: reference attention over
another sequence's stored K with a per-entry rotated Q, plus a KV capacity
decoupled from the logical context. None of that exists upstream, and no
function-level hook can create it, so this build cannot run a RERoT episode.

The contract pinned here is the honest one: refuse, name every missing
capability, and refuse before a model is loaded. It is a regression test for
the failure mode that matters -- a later change that makes the program "run"
by quietly doing ordinary attention, or by pretending a zero-filled capacity is
a measurement. Either would produce plausible numbers under a name that promises
something else.

No model is required: the gate runs in native_execution's constructor.
"""

import argparse
import subprocess
import sys

# Every capability the gate must name. A subset means one of them regressed into
# "works without saying so".
REQUIRED = [
    "reader attention views",
    "whiteboard slots",
    "decoupled KV capacity",
    "batch-invariant matmul",
    "recurrent slot retire",
    "compute release",
]


def run(binary, args):
    completed = subprocess.run([binary, *args], capture_output=True, text=True)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()

    failures = []

    # Help must keep working: the gate is on execution, not on describing the
    # program.
    rc, out = run(args.binary, ["--help"])
    if rc != 0:
        failures.append(f"--help must exit 0, got {rc}\n{out}")
    if "--total-kv" not in out:
        failures.append(f"--help must still document --total-kv\n{out}")

    # A complete command line, so the failure cannot be a parse error. The model
    # path does not need to exist: the gate runs before model loading.
    invocation = [
        "--model", "/nonexistent/herd-rerot-test.gguf",
        "--prompt", "hello world",
        "--stage", "base=answer briefly",
        "--ctx-size", "64",
    ]
    rc, out = run(args.binary, invocation)
    if rc == 0:
        failures.append(f"a RERoT run must fail while the reader station is absent, got exit 0\n{out}")
    for capability in REQUIRED:
        if capability not in out:
            failures.append(f"the refusal must name '{capability}'\n{out}")
    if "reader station" not in out and "station" not in out:
        failures.append(f"the refusal must say what is missing, not only that something is\n{out}")
    # It must fail on capabilities, not on the absent file: a run that got as far
    # as opening the model would mean the gate moved.
    if "could not load the model" in out or "no such file" in out.lower():
        failures.append(f"the gate must run before model loading\n{out}")

    # Decoupled capacity is a separate refusal with its own message, and it must
    # not be reachable by silently rounding --total-kv down to --ctx-size.
    rc, out = run(args.binary, [*invocation, "--total-kv", "128"])
    if rc == 0:
        failures.append(f"--total-kv 128 with --ctx-size 64 must not run\n{out}")

    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        print(f"\n{len(failures)} contract violation(s)", file=sys.stderr)
        return 1
    print("rerot-compose refuses exactly what it cannot do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
