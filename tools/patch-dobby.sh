#!/bin/sh
# Reproducible fixups for the pinned Dobby checkout.
#
# Dobby (jmpews/Dobby @ 5dfc854, == refs/tags/latest) does not compile on a
# 2026 Linux toolchain and its CMake mixes both architectures' assembly into
# one target. Each item below is a defect in the dependency, not a change in
# behaviour; the alternatives were a fork of Dobby (a second upstream to track)
# or these five mechanical edits, applied every time a clean tree is fetched.
#
# Run by FetchContent as PATCH_COMMAND, so a fresh build directory reproduces
# the working dependency without manual steps.
#
#   1. RuntimeModule::load_address was renamed to ::base; two call sites kept
#      the old name.
#   2. MemRange::start became a method, so comparisons need `start()`.
#   3. core/arch/Cpu.h was deleted upstream, but one file still includes it.
#   4. glibc 2.4x hides gettimeofday unless <sys/time.h> is included.
#   5. The closure bridges are GAS-with-cpp (`.intel_syntax noprefix`) but are
#      named `.asm`, which gcc treats as a linker input on Linux; CMake then
#      "compiles" them with `cc`, produces nothing, and `ar` fails. Name them
#      by the dialect they are in so CMake routes them to the assembler and
#      puts `-x assembler-with-cpp` in front. Both architectures are kept in the
#      list: the files gate themselves on __x86_64__/__aarch64__ and the unused
#      one assembles to an empty object.
#   6. get_process_map_with_proc_maps() reads every region as an ELF header
#      before it looks at the mapping name; a non-file-backed mapping ([vvar])
#      then faults on its first byte with SIGBUS, which kills the symbol
#      resolver -- the only route to a file-local function. Reordered in
#      patch-dobby-process-map.py, which asserts its anchor exactly once.
#
# Everything that is not a dependency defect (hook flags, forced includes for
# upstream) lives in the project CMakeLists and never touches third_party.
set -e

sed -i 's/\.load_address/.base/g' \
    source/Backend/UserMode/PlatformUtil/Linux/ProcessRuntime.cc \
    builtin-plugin/SymbolResolver/elf/dobby_symbol_resolver.cc

sed -i 's/a\.start < b\.start/a.start() < b.start()/' \
    source/Backend/UserMode/PlatformUtil/Linux/ProcessRuntime.cc

sed -i '/#include "core\/arch\/Cpu.h"/d' \
    source/Backend/UserMode/ExecMemory/code-patch-tool-posix.cc

sed -i '1i #include <sys/time.h>' external/logging/logging.cc

# 6. The /proc/self/maps reader reads every region as an ELF header before it
#    looks at the mapping name, so [vvar] kills it with SIGBUS and the symbol
#    resolver -- the only route to a file-local function -- is unusable.
#    A cross-block reorder is too fragile for sed, so it lives in its own file
#    that asserts its anchor exists exactly once.
python3 "${0%/*}/patch-dobby-process-map.py"

# Rename the asm sources to `.S` (preprocessed assembly) and let the existing
# globs/EXPLICIT lists pick them up.
sed -i 's/\.asm$/.S/' CMakeLists.txt
for f in $(find source -name '*.asm'); do mv "$f" "${f%.asm}.S"; done
