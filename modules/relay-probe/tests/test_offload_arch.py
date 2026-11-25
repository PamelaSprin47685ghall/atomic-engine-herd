#!/usr/bin/env python3
"""Assert the probe binary carries a code object for every local GPU architecture.

The build succeeding says nothing about which GPUs the fatbin targets: HIP_ARCHITECTURES can
name a card that is not installed, and the binary still links and runs (on nothing useful).
This reads the offloading bundles back out of the executable and requires each architecture
that rocm_agent_enumerator reports locally to appear.

Usage: test_offload_arch.py <llvm-objdump> <binary> <arch> [<arch> ...]
"""
import re
import subprocess
import sys


def offload_archs(objdump, binary):
    out = subprocess.run([objdump, "--offloading", binary],
                         capture_output=True, text=True, check=True).stdout
    return set(re.findall(r"--(gfx[0-9a-z]+)", out))


def main(argv):
    if len(argv) < 4:
        print("usage: test_offload_arch.py <llvm-objdump> <binary> <arch> [<arch> ...]",
              file=sys.stderr)
        return 2
    objdump, binary, required = argv[1], argv[2], argv[3:]
    present = offload_archs(objdump, binary)
    print(f"offload bundles in {binary}: {sorted(present) or '(none)'}")
    missing = [arch for arch in required if arch not in present]
    if missing:
        print(f"FAIL: no code object for {missing} (local GPU architectures)", file=sys.stderr)
        return 1
    print(f"OK: code objects cover {required}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
