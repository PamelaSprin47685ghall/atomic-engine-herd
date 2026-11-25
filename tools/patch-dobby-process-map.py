#!/usr/bin/env python3
"""Reorder the ELF magic check after the mapping-name test in Dobby.

Why: get_process_map_with_proc_maps() walks /proc/self/maps and reads the first
bytes of every readable-private and executable-private region as an ELF header.
Some regions are not backed by file contents -- [vvar] is the common one -- and
touching them raises SIGBUS. There is no return value to check and no errno: the
process dies inside memcmp. Checking the mapping name first is therefore not an
optimisation, it is the only correct order.

Applied by tools/patch-dobby.sh from FetchContent's PATCH_COMMAND, so a fresh
build directory reproduces a working dependency. Idempotent: a tree that already
carries the reorder is left alone.
"""
import sys

PATH = "source/Backend/UserMode/PlatformUtil/Linux/ProcessRuntime.cc"

OLD = """    // check elf magic number
    ElfW(Ehdr) *header = (ElfW(Ehdr) *)region_start;
    if (memcmp(header->e_ident, ELFMAG, SELFMAG) != 0) {
      continue;
    }

    char *path_buffer = line_buffer + path_index;
    if (*path_buffer == 0 || *path_buffer == '\\n' || *path_buffer == '[')
      continue;
"""

NEW = """    // Name test first: a mapping that is not backed by file contents (such as
    // [vvar]) faults on its first byte. SIGBUS, no return value, no errno.
    char *path_buffer = line_buffer + path_index;
    if (*path_buffer == 0 || *path_buffer == '\\n' || *path_buffer == '[')
      continue;

    // check elf magic number
    ElfW(Ehdr) *header = (ElfW(Ehdr) *)region_start;
    if (memcmp(header->e_ident, ELFMAG, SELFMAG) != 0) {
      continue;
    }
"""


def main() -> int:
    with open(PATH, encoding="utf-8") as handle:
        text = handle.read()

    if NEW in text:
        return 0

    count = text.count(OLD)
    if count != 1:
        sys.stderr.write(
            f"patch-dobby-process-map: expected exactly one ELF magic block in {PATH}, "
            f"found {count}; the pinned Dobby revision changed, recheck the patch\n"
        )
        return 1

    with open(PATH, "w", encoding="utf-8") as handle:
        handle.write(text.replace(OLD, NEW))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
