# atomic-engine-herd — clean-room overlay on a pinned upstream.
#
# Two rules this file exists to enforce:
#   1. third_party/upstream is read-only. Nothing here edits it; CMake only
#      *adds compile options* to selected upstream translation units.
#   2. Every behaviour change is a function-level seam installed by Dobby at
#      startup, or a self-contained library we own. No forked source tree.
#
# Layout
#   third_party/upstream   git submodule, pinned commit, never modified
#   patches/               seams (Dobby hooks) + patch runtime
#   libs/                  our own cohesive libraries (added later, per module)
#   tools/                 our own drivers (added later, per module)
cmake_minimum_required(VERSION 3.21)
project(atomic-engine-herd VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(HERD_BUILD_PROBE "Build herd-probe, the seam smoke test" ON)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Dobby: pinned by commit, static, near-branch trampoline + symbol resolver.
# NearBranch is what lets us patch short upstream functions; the resolver is
# what lets us reach file-local (static / anonymous-namespace) upstream code.
# ---------------------------------------------------------------------------
include(FetchContent)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE) # Dobby still says 3.5
# Dobby's master renamed RuntimeModule::load_address to ::base but left the
# Linux module-map reader on the old name; the one-line patch keeps the pin
# buildable without maintaining a fork of the dependency.
# Dobby master (== refs/tags/latest) does not compile on a 2026 toolchain and
# ships arm64/x64 .asm the GNU assembler cannot read. All of it is dependency
# fixups applied at fetch time, so a clean configure reproduces the build and
# nothing in third_party/upstream is involved.
#   1. RuntimeModule::load_address was renamed to ::base; call sites kept the old name
#   2. MemRange::start became a method
#   3. core/arch/Cpu.h was deleted but still included
#   4. glibc 2.4x needs <sys/time.h> for gettimeofday
#   5. drop the .asm trampoline sources; the C++ closure bridge covers x86_64
# A fresh build directory fetches Dobby over the network, and that is the step that
# looks like a hang: a configure that has already downloaded everything still waits
# on the git clone here, measured at minutes on a slow link while the rest of the
# configure takes about a second. Offline or repeat work should point at an existing
# checkout instead of cloning again:
#
#   cmake -S . -B <dir> -DFETCHCONTENT_SOURCE_DIR_DOBBY=$PWD/build/_deps/dobby-src
#
# Note that this bypasses PATCH_COMMAND: the tree must already carry the fixups from
# tools/patch-dobby.sh, which any tree produced by a normal configure does.
FetchContent_Declare(dobby
    GIT_REPOSITORY https://github.com/jmpews/Dobby
    GIT_TAG        5dfc8546954ce3b3198132ab13fddb89ee92cdd7
    GIT_SHALLOW    FALSE
    PATCH_COMMAND  sh "${CMAKE_CURRENT_SOURCE_DIR}/tools/patch-dobby.sh")

set(DOBBY_GENERATE_SHARED   OFF CACHE BOOL "" FORCE)
set(DOBBY_DEBUG             OFF CACHE BOOL "" FORCE)
set(NearBranch              ON  CACHE BOOL "" FORCE)
set(Plugin.SymbolResolver   ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(dobby)

# ---------------------------------------------------------------------------
# Upstream, as a pinned submodule. Static build: the seams live in the same
# image as the functions they patch, so addresses are link-time facts and no
# dlsym/RTLD lookup or exported-symbol list is involved.
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP     OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL       OFF CACHE BOOL "" FORCE)
# No LTO: cross-TU inlining would let upstream inline a hooked function into
# its callers, leaving the seam installed on a body nobody calls any more.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_subdirectory(third_party/upstream EXCLUDE_FROM_ALL)

# Older comment kept for history: the rename fix below supersedes it.
# Newer libstdc++ stopped pulling <system_error>/<stdexcept> in transitively and
# two upstream headers rely on them (llama-prefill-wide.h catches
# std::system_error, llama-prefill-window-worker.h throws). A forced include is
# a compile option, not an edit: the submodule stays byte-identical.
# SHELL: keeps "-include system_error" one argv pair; without it CMake splits
# on the semicolon and the compiler sees `stdexcept` as a linker input.
target_compile_options(llama PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include system_error>"
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include stdexcept>")

# ---------------------------------------------------------------------------
# Patchable translation units.
#
# -fpatchable-function-entry=16 puts 16 bytes of NOPs before every function in
# the TU: Dobby's absolute jump lands there, so short functions and functions
# with awkward prologues stop being a problem.
# -fno-inline / -fno-ipa-* stop GCC from cloning or inlining the functions we
# address (&Foo::bar must name a body, not a hint).
# -fcf-protection=none drops endbr64, which otherwise turns an indirect jump
# from the trampoline into SIGSEGV on CET-enabled CPUs.
#
# Add a path here when a seam targets a function defined in it. Nothing else
# about the upstream build changes.
# ---------------------------------------------------------------------------
# One entry per translation unit that defines a hooked function. A seam on a
# function from a TU not listed here still installs, but the compiler may inline
# the body into that TU's callers, leaving the hook on an address nobody calls.
set(HERD_HOOK_TUS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src/llama.cpp
    # ggml_backend_sched_graph_compute{,_async}: the scheduler-capture seams that
    # let herd::phase learn a context's scheduler without reading a private field.
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/src/ggml-backend.cpp
    # llama_free: the binding must die with the context, or a later context that
    # reuses the address inherits a dangling scheduler pointer.
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src/llama-context.cpp
    # llm_graph_context::build_attn_mha -- the READER's real seam. build_attn, which the
    # reader lane used to hook, is the wrong boundary: between its entry and the attention
    # math it writes the KV cache (cpy_k/cpy_v), and between the math and its return it
    # applies build_attn_scale_kv_output, an optional Hadamard rotation (self_v_rot), the
    # LoRA output projection and the output bias. A replacement at build_attn would owe an
    # identical copy of all of that; a replacement at build_attn_mha owes only the
    # [n_embd_head_q, n_head_q, n_tokens] attention result, which is what the emitter
    # produces. Hooked here as a probe first (HERD_READER_MHA), because the seam's shape
    # has to be measured before anything is replaced at it.
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src/llama-graph.cpp
    # DELIBERATELY ABSENT: src/llama-model.cpp, which defines
    # llama_model::create_memory -- the function the decoupled-KV hook patches
    # (patches/herd_nkv.cpp). This is a reasoned exclusion, not an oversight, and it
    # rests on two facts that a pin bump could change:
    #
    #   1. Its only caller is llama-context.cpp:602, a DIFFERENT translation unit, and
    #      INTERPROCEDURAL_OPTIMIZATION is OFF, so the body cannot be inlined into the
    #      call site. The hook is therefore on every path that reaches it.
    #   2. There is exactly one definition in the object and no clone. Verified:
    #        nm build/third_party/upstream/src/CMakeFiles/llama.dir/llama-model.cpp.o \
    #          | grep 'create_memory.*llama_memory_params'
    #      shows one `T` symbol (plus its own `.cold` section, which is part of the same
    #      function). A `.constprop`, `.isra` or `.part` variant would mean the compiler
    #      split the body and a call path could bypass the hook -- at which point this TU
    #      must be added here and rebuilt.
    #
    # The cost of adding it unconditionally is real: this is the largest, hottest TU in
    # the library and every function in it would lose inlining. So the exclusion is
    # bought with the check above rather than with flags: re-run it on every pin bump.
    # A short prologue is not a silent failure -- DobbyHook returns non-zero and
    # herd::install_all() aborts -- but a clone would be, and only that check catches it.
)

# The compute-release experiment hooks six allocator entry points, two of which
# (ggml_gallocr_get_buffer_size, ggml_gallocr_free) are short enough that a hook
# cannot land without the padded prologue. Measured: without this, the installer
# aborts with "DobbyHook failed" for those symbols. The flag is conditional on the
# experiment precisely because it costs inlining in a hot TU.
# ggml-alloc.c is padded unconditionally now, and the reason is architectural rather than
# convenient: the compute release cannot free an allocator it cannot name, and the only way
# to name it from outside upstream is the ggml_gallocr_new_n seam -- whose target lives in
# this file. Leaving the file gated meant a product could call release_and_arm() while the
# hook that makes it possible was not compiled, which is exactly the undefined symbol the
# linker reported. The seam is the mechanism, so the mechanism's TU is in the pad set.
list(APPEND HERD_HOOK_TUS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/src/ggml-alloc.c)

set(_herd_upstream_dir ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream)

if(MSVC)
    set(HERD_HOOK_FLAGS "/Ob0;/hotpatch")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(HERD_HOOK_FLAGS -fno-inline -fpatchable-function-entry=4 -fcf-protection=none)
else()
    set(HERD_HOOK_FLAGS -fno-inline -fpatchable-function-entry=16 -fcf-protection=none)
endif()

if(NOT MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND HERD_HOOK_FLAGS -fno-ipa-cp -fno-ipa-sra -fno-ipa-icf)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND HERD_HOOK_FLAGS -fno-inline-functions -fno-lto)
    endif()
endif()

# Source-file properties are directory-scoped, and the scope that counts is the
# one that calls add_library(): llama.cpp is compiled by upstream's src/
# directory, not by the project directory and not by the submodule root. The
# first version of this file set the property in the project directory (silently
# ignored); the second set it on the submodule root (also ignored). Both left
# every seam installed on an unpadded, inlinable body with no error reported.
# DIRECTORY names the owning scope (CMake >= 3.18).
set_property(SOURCE ${HERD_HOOK_TUS} DIRECTORY ${_herd_upstream_dir}/src
             PROPERTY COMPILE_OPTIONS "${HERD_HOOK_FLAGS}")

# ---------------------------------------------------------------------------
# Seams. OBJECT, never STATIC: a static archive whose only role is a static
# initializer would be dropped by the linker because nobody references it.
# ---------------------------------------------------------------------------
# The attention seam: opt-in at runtime (HERD_READER_ATTN), transparent when the
# lane is off, and a hard stop when the lane is on. Compiled in unconditionally
# because an unused hook costs one table entry, not a code path -- and because a
# seam that only exists in some configurations is a seam nobody tests.
set(HERD_READER_ATTN_SOURCE patches/herd_reader_attn.cpp)

add_library(herd_seams OBJECT
    patches/herd_patch.cpp
    ${HERD_READER_ATTN_SOURCE}
    # The one translation unit that reads llm_graph_context's private ctx0. Isolated on
    # purpose: it is the only place that must recompile when upstream's graph header moves,
    # and everything else sees `struct ggml_context` and one function.
    patches/herd_graph_access.cpp
    # The K/V cache read: inp->mctx->get_k/get_v, three public symbols and no copied body.
    # Its own translation unit because it is the second (and last) file that includes an
    # upstream private header, and a header change there should recompile exactly two files.
    patches/herd_graph_cache.cpp
    patches/herd_phase.cpp
    # The station itself: no ggml, no llama, no Dobby. Split out so it can be
    # tested with fabricated pointers instead of a model.
    patches/herd_phase_station.cpp
    # The release mechanism, not an experiment: release_and_arm() is what
    # --phase-memory calls, and the allocator hooks in this TU are how it finds
    # the allocator to free. It used to sit behind HERD_BUILD_BUFFER_HOOKS while
    # the product called into it, so a default build failed to link -- the gate
    # and the caller disagreed about whether the feature existed.
    patches/herd_phase_buffers.cpp
    patches/herd_prefab.cpp
    patches/herd_prefab_station.cpp
    # Decoupled KV capacity: inert unless a driver opens a herd::nkv::scope around one
    # init call. Linked here so --total-kv has a mechanism; it reads no environment, so
    # no configuration can leak from one context to the next.
    patches/herd_nkv.cpp
    # The ruling and its outcome recording, split out so the failure paths (upstream
    # returns null / throws) are testable without a model. `name(refusal)` lives here too,
    # so this is a link-time dependency of the hook above.
    patches/herd_nkv_decision.cpp
    # The model's memory shape: reads `model->arch` through upstream's private header and
    # states the fork's refusal rule itself, instead of inheriting it from a membership
    # predicate that happens to list DEEPSEEK4 today.
    patches/herd_model_arch.cpp
    # The allocator station's bookkeeping. Pure: no ggml type, no hook. Linked so
    # its test can link it; the hooks that would use it are the separate
    # herd_buffer_hooks target below, which is research and linked into nothing.
    patches/herd_buffer_station.cpp)
target_include_directories(herd_seams PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/patches)
# The prefab seam builds a padded llama_ubatch, so it needs that type's definition.
# Upstream's private headers are included, never edited; a pin bump that renames a
# field fails this build, which is the wanted direction of failure.
target_include_directories(herd_seams PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src)
# NOTE: there was a dependency here on `tq_reader_graph`, added for the row-table publisher. It is
# gone with the publisher (see the note in `patches/herd_reader_attn.cpp`), and removing it was not
# only cleanup: `tq_reader_graph` is created inside `modules/reader/CMakeLists.txt`, which lives
# under `if(HERD_BUILD_MODULES)` and further under `if(BUILD_TESTING)` and
# `if(TARGET ggml AND TARGET ggml-cpu)`. Linking it from `herd_seams`, which is defined before
# `add_subdirectory(modules)` runs, made every configuration with `-DHERD_BUILD_MODULES=OFF` or
# `-DBUILD_TESTING=OFF` fail at generate time with an unknown target. A patch layer must not depend
# on an optional module target, so the coupling is not restored until something in the patch layer
# genuinely needs it -- and then it has to be guarded on `if(TARGET tq_reader_graph)`.
target_link_libraries(herd_seams PUBLIC llama ggml)
target_compile_options(herd_seams PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none>)
# The release path is compiled in, so the guard that selects it must say so in one place
# that cannot drift from the source list above. It used to be implied by an option that
# also decided whether the TU was linked -- two facts, one switch, and they disagreed.
target_compile_definitions(herd_seams PUBLIC HERD_RELEASE_AVAILABLE=1)
# The hook layer is the most safety-critical code in the tree -- it runs inside upstream's
# own calls and its mistakes are silent -- and until now it was the only part compiled
# without the diagnostics every station test gets. A `switch` over a refusal enum that
# gained a value, or a comparison across two unrelated pointer types, would have shipped
# unnoticed. Same flags as the tests, so "it compiles" means the same thing there.
#
# -Wno-deprecated-declarations is NOT added: these files must not use the deprecated
# llama entry points, and if one starts to, the warning is worth reading.
#   herd_nkv.cpp / herd_model_arch.cpp: no llama_* calls at all beyond the hook target.
#   herd_prefab.cpp: reads llama_ubatch fields only.
target_compile_options(herd_seams PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
if(HERD_SEAMS_WERROR)
    target_compile_options(herd_seams PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Werror>)
endif()
target_link_libraries(herd_seams PUBLIC dobby_static)

# Consumers must link `herd_seams`, not just `llama`. Linking an OBJECT library
# adds its objects to the consumer (CMake >= 3.12), which is what carries the
# static registrars into the image. An executable that links only `llama` gets
# no seams and no error: register-time code never runs, install_all() installs
# nothing, and the engine looks patched because nobody checked.
add_library(herd::seams ALIAS herd_seams)

# ---------------------------------------------------------------------------
# Smoke test: the same seam runtime, compiled with its probe main. Proves that
# a hook is installed before use, that upstream still runs underneath it, and
# that the replacement is observed from ordinary calling code.
# ---------------------------------------------------------------------------
if(HERD_BUILD_PROBE)
    add_executable(herd-probe patches/herd_patch.cpp)
    target_compile_definitions(herd-probe PRIVATE HERD_PROBE_MAIN=1)
    target_include_directories(herd-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/patches)
    target_compile_options(herd-probe PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none>)
    target_link_libraries(herd-probe PRIVATE llama dobby_static ${CMAKE_DL_LIBS} Threads::Threads)

    # The registry under real concurrency, with real hooks.
    #
    # Not folded into herd-probe: that binary installs from its own constructor, and this one
    # needs the registry still open when main() starts, so it runs with HERD_NO_AUTOLOAD=1.
    # Separate target also means a race in the installer fails one named test instead of
    # taking the smoke test with it.
    add_executable(herd-registry-test patches/herd_registry_test.cpp)
    target_include_directories(herd-registry-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/patches)
    target_compile_options(herd-registry-test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none>)
    target_link_libraries(herd-registry-test PRIVATE herd_seams llama dobby_static ${CMAKE_DL_LIBS} Threads::Threads)
endif()

# Station tests: the isolation boundary the seam depends on, exercised without a
# model, a backend, or a patched function.
add_executable(herd-phase-station-test patches/herd_phase_station_test.cpp
    patches/herd_phase_station.cpp)
target_include_directories(herd-phase-station-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/patches)
target_compile_options(herd-phase-station-test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
target_link_libraries(herd-phase-station-test PRIVATE Threads::Threads)

# The release rule, on configurations this host cannot build (one GPU backend, two buffer
# types, two CPU backends sharing a type). Links only the pure station TU: no ggml, no
# llama, no Dobby, so the table can be walked on any machine.
add_executable(herd-release-decision-test patches/herd_release_decision_test.cpp
    patches/herd_phase_station.cpp)
target_include_directories(herd-release-decision-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/patches)
target_compile_options(herd-release-decision-test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
add_test(NAME herd.release_decision COMMAND herd-release-decision-test)

# The cache probe's arithmetic, on the two cases that would otherwise end the process.
#
# Not a reader test and not gated on a model, because the rule is pure: free bytes, per-view
# cost, how many views. What it pins is that a view which cannot fit is REFUSED rather than
# asked for -- ggml_new_tensor_impl asserts on an overfull arena in every build (ggml.c:1832),
# so the question "is there room?" must be answered before ggml is called, not after it returns.
#
# It links herd_graph_cache.cpp because cache_headroom_ok lives beside cache_tensors, which is
# what keeps the rule and its single caller in one translation unit. That costs the llama include
# path; it costs no model, no context and no device.
add_executable(herd-cache-headroom-test
    patches/herd_cache_headroom_test.cpp
    patches/herd_graph_cache.cpp)
target_include_directories(herd-cache-headroom-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/patches
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src)
target_compile_options(herd-cache-headroom-test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
target_link_libraries(herd-cache-headroom-test PRIVATE llama ggml)
add_test(NAME herd.cache_headroom COMMAND herd-cache-headroom-test)

# The model-memory classification, pinned over every arch this pin defines. It includes
# upstream's `llama-arch.h` read-only: hard-coded integer values would pass after a pin bump
# renumbers the enum, and then pin nothing. No model, no GGUF, no llama link.
add_executable(herd-model-arch-test patches/herd_model_arch_test.cpp
    patches/herd_model_arch.cpp)
target_include_directories(herd-model-arch-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/patches
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/include)
target_compile_options(herd-model-arch-test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
# It links llama for one reason: llm_arch_is_recurrent / llm_arch_is_hybrid are defined in
# upstream's llama-arch.cpp, and the test asserts the classification *against* those same
# predicates. Re-implementing them in the test would have made it a second copy of the rule
# under test -- the exact shape that let an earlier mutation survive.
target_link_libraries(herd-model-arch-test PRIVATE llama)
add_test(NAME herd.model_arch COMMAND herd-model-arch-test)

# The decoupled-capacity ruling, tested without a model. It carries the one claim a
# model-backed probe cannot state: a legal request that upstream fails to deliver
# (null, or a throw out of its constructors) is never recorded as applied.
add_executable(herd-nkv-decision-test
    patches/herd_nkv_decision_test.cpp
    patches/herd_nkv_decision.cpp)
target_include_directories(herd-nkv-decision-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/patches)
target_compile_options(herd-nkv-decision-test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
add_test(NAME herd.nkv_decision COMMAND herd-nkv-decision-test)

enable_testing()
if(HERD_BUILD_PROBE)
    add_test(NAME herd.seams COMMAND herd-probe)
    add_test(NAME herd.registry COMMAND herd-registry-test)
    set_tests_properties(herd.registry PROPERTIES ENVIRONMENT "HERD_NO_AUTOLOAD=1")

    # What an explicit reader request does when the build cannot honour it.
    #
    # Four environment combinations against the real product binary, because the requirement is
    # stated on what a user sees: exit status and the two streams. The two --help cases need no
    # model and guard the constructor, which is where the previous defect lived (a request that
    # was answered only under HERD_SEAM_TRACE, i.e. silently in ordinary use).
    #
    # The model is optional and the test says so when it is absent, rather than reporting a pass
    # over cases it skipped. Registering the model here would duplicate the other model-gated
    # options; passing none is the honest default, and the option below is how a full run is asked
    # for.
    add_executable(herd-reader-request-test patches/herd_reader_request_test.cpp)
    target_compile_options(herd-reader-request-test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
    set(HERD_READER_REQUEST_TEST_MODEL "" CACHE FILEPATH
        "Real GGUF model for herd.reader_request (enables the two attention-reaching cases).")
    # NOTE: this variable is READ at the end of the file (after `add_subdirectory(modules)`), so it
    # cannot default to `TQ_TINY_LLAMA_MODEL` here -- that variable is defined later, just before
    # `option(HERD_BUILD_MODULES)`. The default is applied at the registration site; see the
    # `herd.reader_request` block.

    # THE REGISTRATION IS AT THE END OF THIS FILE, NOT HERE, AND THE REASON IS A TRAP THIS TREE HAS
    # RECORDED TWICE ALREADY: the target it names is defined in `modules/`, and this block runs
    # before `add_subdirectory(modules)`, so `if(TARGET ...)` here is FALSE even with modules on --
    # invisibly: no error, no test, just an absence.
    #
    # THE SUBJECT IS `llama-corpus`, AND ONE REVISION GOT THAT WRONG BY READING THE USAGE LINE.
    # The usage line says `<product-binary>`, which invites the reading "any product", and the last
    # revision took it as `tq_batch_compose`. Measured, both ways:
    #
    #   tq-batch-compose score -m M --corpus D   -> "tq-batch-compose: unknown option: score"
    #   llama-corpus     score -m M --corpus D   -> loads the model and proceeds
    #
    # Cases 4-5 of `herd_reader_request_test` run exactly `score -m MODEL --corpus DIR`, and that
    # subcommand exists in ONE binary: `llama-corpus`. Its name says "corpus", but it both BUILDS a
    # corpus and SCORES one (see its own `--help`), and the test uses the second function. So the
    # binary in that slot is the tool that can reach attention under `score`, not the generation
    # product -- and the attention cases were not "impossible in any build", as the previous
    # revision claimed; they were merely never given a model and a corpus.
    #
    # What stays here is the executable, unconditional: it links the runtime and llama only.
endif()
add_test(NAME herd.phase_station COMMAND herd-phase-station-test)

# ---------------------------------------------------------------------------
# Decoupled KV capacity: an experiment, linked into no product.
#
# It hooks llama_model::create_memory to hand the memory module a copied cparams with a
# larger n_ctx_seq, so the cell pool grows while the context's logical window, rope
# selection and bounds checks keep reading its own (unchanged) copy. Whether that is
# really decoupled capacity -- as opposed to a bigger --ctx-size in disguise -- is decided
# by herd-nkv-probe: same tokens, same positions, bit-identical logits, larger state, and
# the window still reported unchanged.
#
# OFF by default, for three reasons -- the first two are why the sibling experiment
# (HERD_BUILD_BUFFER_HOOKS below) is OFF as well, and the third is specific to this one:
#
#   1. Its central claim is unproven. On the one model measured, llama_state_get_size()
#      was identical with and without the hook, so "the cell pool actually grew" has no
#      observation behind it. A default build should contain what works, not what is
#      being investigated.
#   2. It includes two upstream PRIVATE headers (llama-cparams.h, llama-memory.h) to name
#      the hooked signature's types. That is the right trade for an experiment -- a pin
#      bump that moves a field fails the build instead of misinterpreting memory -- but it
#      is a coupling no default build should carry for a feature no product links.
#   3. The alternative justification, "compile it by default so it is rot-checked against
#      every pin bump", is real but weaker than the cost: the same check is a single
#      -DHERD_BUILD_NKV=ON away, and an experiment that cannot ship does not need to be
#      green in every default build to stay honest.
#
# Nothing here is deleted: the sources, the probe and its measurement stay in the tree.
# ---------------------------------------------------------------------------
option(HERD_BUILD_NKV "Build the decoupled-KV experiment and its probe (alpha)" OFF)
# ---------------------------------------------------------------------------
# Envelope padding at the point where ubatches are born.
#
# The lane that padded a ubatch after the fact could never work: decode() asks the
# memory module for a cell plan first, and that plan is built from the ubatch
# (llama_kv_cache::prepare -> find_slot) before process_ubatch runs, so a padded
# ubatch always contradicted its own plan. This seam sits on
# llama_batch_allocr::split_simple, which runs before prepare, so the plan is built
# from the padded shape.
#
# OFF by default: its claim is under test, it includes an upstream private header,
# and no product links it. The probe is the only thing that runs it.
# ---------------------------------------------------------------------------
option(HERD_BUILD_PREFAB_SPLIT "Build the ubatch-envelope experiment and its probe (alpha)" OFF)
if(HERD_BUILD_PREFAB_SPLIT)
    add_executable(herd-prefab-split-probe
        patches/herd_prefab_split_probe.cpp
        patches/herd_prefab_split.cpp
        patches/herd_patch.cpp)
    target_include_directories(herd-prefab-split-probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/patches
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/upstream-ext/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src)
    target_compile_options(herd-prefab-split-probe PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none>)
    # `tq_upstream_ext` is what defines `prefab_register`. Without it the provider below is a
    # declaration with no definition -- a link error, at best; at worst a probe that silently
    # dropped the registration if the call had been made optional.
    target_link_libraries(herd-prefab-split-probe PRIVATE
        llama dobby_static tq_upstream_ext ${CMAKE_DL_LIBS} Threads::Threads)

    # The reservation's own contract, separate from the lane's end-to-end probe: what a
    # scratch_scope promises about the cells it borrows, and what it does when a promise
    # cannot be kept. Registered only with a model, because the interesting states are all
    # reachable only through a real decode -- and a test that skips without one would report
    # green having run nothing.
    # The test needs a GGUF, and where it comes from is stated rather than guessed. Two sources:
    #
    #   * TQ_PREFAB_SPLIT_TEST_MODEL set by the operator -> used verbatim. It is their path in
    #     their directory; writing a model there would overwrite a file this build does not own,
    #     and a cache value pointing at a deleted model must be REPORTED, not silently refilled
    #     (an earlier revision of this file did exactly that, and it generated over whatever the
    #     operator had named).
    #
    #   * unset (the default) -> a fixture this build owns, in the BUILD tree, generated at TEST
    #     time by a CTest setup fixture. Living in the build tree means it survives a /tmp wipe
    #     and never needs a reconfigure; being a fixture means `ctest -R herd.prefab_split_scope`
    #     still works. The test is REGISTERED either way -- previously an empty path meant no test
    #     at all, which made a missing model look like a suite that had nothing to say.
    set(TQ_PREFAB_SPLIT_TEST_MODEL "" CACHE FILEPATH
        "GGUF for herd.prefab_split_scope; empty generates one in the build tree")
    set(_tq_prefab_model "${TQ_PREFAB_SPLIT_TEST_MODEL}")
    set(_tq_prefab_fixture "")
    if(NOT _tq_prefab_model)
        set(_tq_prefab_model "${CMAKE_CURRENT_BINARY_DIR}/tq-prefab-split-tiny.gguf")
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND AND
           EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny-transformer/make_tiny_transformer.py")
            set(_tq_prefab_fixture tq_prefab_split_model_fixture)
            add_test(NAME ${_tq_prefab_fixture}
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny-transformer/make_tiny_transformer.py"
                        "${_tq_prefab_model}")
            # SETUP, not a default test: it runs before the test that requires it, and only when
            # that test is selected. A fixture that ran on every `ctest` invocation would make an
            # unrelated suite pay a generation it never reads.
            set_tests_properties(${_tq_prefab_fixture}
                PROPERTIES FIXTURES_SETUP tq_prefab_split_model)
        else()
            message(WARNING
                "herd.prefab_split_scope: no python3 to generate ${_tq_prefab_model}. The test is "
                "still registered and will fail at model load until you pass "
                "-DTQ_PREFAB_SPLIT_TEST_MODEL=<path to a GGUF>; it will not report a verdict it "
                "did not reach.")
        endif()
    endif()
    # Registered unconditionally, because the model now always has a home: either the operator's
    # path or the build-tree fixture. The guard used to be `if(TQ_PREFAB_SPLIT_TEST_MODEL)`, which
    # meant the DEFAULT configuration silently had no test at all.
    add_executable(herd-prefab-split-scope-test
            patches/herd_prefab_split_scope_test.cpp
            patches/herd_prefab_split.cpp
            patches/herd_patch.cpp)
        target_include_directories(herd-prefab-split-scope-test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/patches
            ${CMAKE_CURRENT_SOURCE_DIR}/modules/upstream-ext/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src)
        target_compile_options(herd-prefab-split-scope-test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none>)
        target_link_libraries(herd-prefab-split-scope-test PRIVATE
            llama dobby_static tq_upstream_ext ${CMAKE_DL_LIBS} Threads::Threads)
        # The seam is opt-in at run time, so the test's environment says so; without it the
        # binary refuses at the door instead of failing five assertions about absent state.
        add_test(NAME herd.prefab_split_scope
            COMMAND ${CMAKE_COMMAND} -E env HERD_PREFAB_SPLIT=1
                    $<TARGET_FILE:herd-prefab-split-scope-test>
                    "${_tq_prefab_model}")
        # Paired with the setup fixture above, and only when one exists: `-R` selection still
        # generates the model first, and a rebuild after a build-tree wipe needs no reconfigure.
        if(_tq_prefab_fixture)
            set_tests_properties(herd.prefab_split_scope
                PROPERTIES FIXTURES_REQUIRED tq_prefab_split_model)
        endif()

    # The reader replacement switch is fail-closed only if a run that asks for it can never exit 0.
    # That is a claim about exit codes and about stdout, so it is checked by running the product
    # binary (see tests/replace-failclosed-test.sh) rather than by reading the source. Case 3 needs
    # a real process for a reason: the refusing code sits inside build_attn_mha, so a run whose
    # attention is never built would never enter it, and only an exit-time check can notice.
    #
    # It reuses this fixture because it needs a model that actually decodes; the switch has nothing
    # else to do with the prefab lane.
    add_test(NAME tq_reader.replace_failclosed
        COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/tests/replace-failclosed-test.sh"
                $<TARGET_FILE:tq_batch_compose> "${_tq_prefab_model}")
    if(_tq_prefab_fixture)
        set_tests_properties(tq_reader.replace_failclosed
            PROPERTIES FIXTURES_REQUIRED tq_prefab_split_model)
    endif()
endif()

if(HERD_BUILD_NKV)
    add_executable(herd-nkv-probe
        patches/herd_nkv_probe.cpp
        patches/herd_nkv.cpp
        patches/herd_nkv_decision.cpp
        patches/herd_model_arch.cpp
        patches/herd_patch.cpp)
    target_include_directories(herd-nkv-probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/patches
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/include
        # llama-cparams.h / llama-memory.h live beside the sources they belong to.
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src)
    target_compile_options(herd-nkv-probe PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none>)
    target_link_libraries(herd-nkv-probe PRIVATE llama dobby_static ${CMAKE_DL_LIBS} Threads::Threads)
endif()

# The prefab lane's rulings: which ubatches may be padded, what happens at the
# ladder's edges, and that a decline is counted under its own reason.
add_executable(herd-prefab-station-test patches/herd_prefab_station_test.cpp
    patches/herd_prefab_station.cpp)
target_include_directories(herd-prefab-station-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/patches)
target_compile_options(herd-prefab-station-test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
# Lets the test place the reservation-id counter so its 2^32 wraparound is exercised in a
# test instead of after four billion calls. Defined on this target only, so the production
# class keeps the same layout in every other translation unit.
target_compile_definitions(herd-prefab-station-test PRIVATE HERD_SCRATCH_POOL_INSPECT=1)
add_test(NAME herd.prefab_station COMMAND herd-prefab-station-test)

# The allocator station's bookkeeping: address reuse, double ownership, and the
# release contract that frees memory exactly once.
add_executable(herd-buffer-station-test patches/herd_buffer_station_test.cpp
    patches/herd_buffer_station.cpp)
target_include_directories(herd-buffer-station-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/patches)
target_compile_options(herd-buffer-station-test PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
add_test(NAME herd.buffer_station COMMAND herd-buffer-station-test)

# ---------------------------------------------------------------------------
# Research, not a feature: the compute-buffer release by allocator redirection.
#
# It compiles, so the idea stays checked against this pin, and it is linked into
# nothing, so no product can run it. The reason is in the file's header and in
# herd::phase::release_unavailable_reason(): llama_context::sched_reserve() returns
# early unless its private `sched_need_reserve` is set, so a release cannot be
# followed by a decode that re-arms the scheduler. On one buffer type the memory
# comes straight back; on more than one the decode fails outright -- and more than
# one is the five-GPU case the feature exists for.
#
# Promoting it needs (a) an upstream way to re-arm the reservation, (b) a lock
# around its tables, and (c) measurement on the multi-device rig. Until then this
# target is OFF and it does not compile as written, which is stated rather than
# hidden: it refers to GGML_SCHED_MAX_BACKENDS (a macro defined inside
# ggml-backend.cpp, not in any header) and to two helpers declared after their use.
# Those are mechanical, and they are not worth fixing for code that cannot ship.
# What the finding turned out to be: upstream's use of a scheduler's allocator is a
# closed set of six public entry points, and ggml_backend_sched_alloc_splits()
# already reserves and retries when an allocation fails -- so the re-reservation
# this feature needs is reachable by a path upstream takes by itself. Nothing has to
# be written into sched_need_reserve, and no extra hook is needed.
#
# So the experiment compiles by default and is still linked into no product: the
# capability is claimed only after herd-release-probe shows the memory leaves, comes
# back, and leaves the logits bit-identical.
# OFF by default, on purpose. Two reasons, both measured:
#   1. the probe currently aborts with heap corruption (malloc(): unsorted double
#      linked list corrupted) in its release path, so a default build would ship a
#      binary that crashes by design-in-progress;
#   2. enabling it adds ggml-alloc.c to HERD_HOOK_TUS, which strips inlining from a
#      hot translation unit for a feature no product links.
# A default build must contain only what works. Turning it on is one flag, and the
# probe is the only thing that prints the numbers the claim needs.
option(HERD_BUILD_BUFFER_HOOKS "Build the compute-release measurement probe" OFF)
if(HERD_BUILD_BUFFER_HOOKS)
    # The hooks themselves are no longer conditional: they are in herd_seams, because the
    # product's --phase-memory path needs them. This option now builds only the probe.
    #
    # The measurement. Needs a model, so it is not a ctest: run it by hand with a
    # GGUF and read the two numbers it prints.
    add_executable(herd-release-probe
        patches/herd_release_probe.cpp
        patches/herd_phase.cpp
        patches/herd_phase_station.cpp
        patches/herd_buffer_station.cpp
        patches/herd_phase_buffers.cpp
        patches/herd_patch.cpp)
    target_include_directories(herd-release-probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/patches
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/include)
    target_compile_options(herd-release-probe PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none>)
    # Same switch the library defines, because this target compiles herd_phase.cpp itself.
    target_compile_definitions(herd-release-probe PRIVATE HERD_RELEASE_AVAILABLE=1)
    target_link_libraries(herd-release-probe PRIVATE llama ggml dobby_static ${CMAKE_DL_LIBS} Threads::Threads)

    # The embeddings attestation, which the release depends on and nothing measured.
    #
    # The release re-arms the scheduler by toggling llama_set_embeddings(), the only public
    # lever that sets sched_need_reserve. Toggling twice restores the value -- but only if the
    # caller's `embeddings_currently` is the context's actual setting, and upstream has no
    # getter. A wrong value does not fail: it silently leaves the context with embeddings
    # INVERTED, and every later llama_get_embeddings() throws. This target measures both
    # directions on a real model, so the requirement is a checked fact rather than a comment.
    #
    # Same source list as the probe: it calls release_and_arm, which is compiled into a
    # translation unit of its own and needs the allocator hooks beside it.
    add_executable(herd-embeddings-attestation-test
        patches/embeddings_attestation_test.cpp
        patches/herd_phase.cpp
        patches/herd_phase_station.cpp
        patches/herd_buffer_station.cpp
        patches/herd_phase_buffers.cpp
        patches/herd_patch.cpp)
    target_include_directories(herd-embeddings-attestation-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/patches
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/ggml/include)
    target_compile_options(herd-embeddings-attestation-test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fcf-protection=none -Wall -Wextra>)
    target_compile_definitions(herd-embeddings-attestation-test PRIVATE HERD_RELEASE_AVAILABLE=1)
    target_link_libraries(herd-embeddings-attestation-test PRIVATE
        llama ggml dobby_static ${CMAKE_DL_LIBS} Threads::Threads)

    # Registered only when a model is offered, which is the same convention the other
    # model-gated tests use (TQ_BATCH_COMPOSE_TEST_MODEL, TQ_COMPOSE_TEST_MODEL): a test that
    # needs a GGUF cannot be a default ctest, and a registration that ran it without one would
    # fail on its own usage message rather than on the property it measures.
    #
    # What it measures, and why it is worth a model: the release re-arms the scheduler by
    # toggling llama_set_embeddings(), and upstream has no getter, so the caller's attestation
    # is unverifiable by the API. The test pins both directions -- a correct attestation
    # preserves embeddings, a wrong one silently disables them -- so the requirement on the
    # caller is a measured fact about the API rather than a sentence in a comment. Case 2
    # asserts the hazard on purpose: if upstream ever gains a getter, case 2 starts failing,
    # and that failure is the signal to delete the parameter.
    set(HERD_EMBEDDINGS_TEST_MODEL "${CMAKE_BINARY_DIR}/fixtures/tiny-transformer.gguf" CACHE FILEPATH
        "Real GGUF model for herd.embeddings_attestation (defaults to the build-tree fixture).")
    # Registered unconditionally, and the model now always has a home. The old `if(...)` guard made
    # the default configuration contain no test at all -- an absence that reads like a pass.
    # The path is written literally rather than through `TQ_TINY_LLAMA_MODEL` because this block
    # runs BEFORE that variable is defined (it is set just above `option(HERD_BUILD_MODULES)`);
    # reading it here would silently expand to an empty string and reintroduce the /tmp-era gap.
    add_test(NAME herd.embeddings_attestation
             COMMAND herd-embeddings-attestation-test "${HERD_EMBEDDINGS_TEST_MODEL}")
    set_tests_properties(herd.embeddings_attestation
        PROPERTIES FIXTURES_REQUIRED tq_tiny_llama)
endif()

# ---------------------------------------------------------------------------
# Our own layer: cohesive libraries and drivers, original code only. They use
# the same in-tree llama/ggml targets the seams patch.
# ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# THE SHARED TINY MODEL FIXTURE, defined ONCE here, before any consumer registers a test.
#
# Why it exists: six model-gated tests used to read /tmp/tiny-*.gguf, files this build never
# created and never owned. A /tmp wipe silently turned them into failures at model load, and a
# machine that happened to have those files passed regardless of their contents. The path is now
# inside the build tree, which survives a /tmp wipe by construction.
#
# It is a CTest SETUP fixture: `ctest -R <one test>` still generates the model first, and a
# build-tree wipe needs no reconfigure. An operator-supplied path still wins -- each consumer
# cache below defaults to this path but keeps its own override.
#
# `TQ_TINY_LLAMA_MODEL` is a plain variable, not a cache entry of its own, so one operator
# override does not have to fight a second cached path.
# -------------------------------------------------------------------------------------------
set(TQ_TINY_LLAMA_MODEL "${CMAKE_BINARY_DIR}/fixtures/tiny-transformer.gguf")
set(TQ_TINY_LLAMA_FIXTURE "")
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny-transformer/make_tiny_transformer.py")
        add_test(NAME tq_tiny_llama_fixture
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/tiny-transformer/make_tiny_transformer.py"
                    "${TQ_TINY_LLAMA_MODEL}")
        set_tests_properties(tq_tiny_llama_fixture PROPERTIES FIXTURES_SETUP tq_tiny_llama)
        set(TQ_TINY_LLAMA_FIXTURE tq_tiny_llama)
    else()
        message(WARNING
            "tq_tiny_llama_fixture: no python3 to generate ${TQ_TINY_LLAMA_MODEL}. Model-gated "
            "tests stay registered and will fail at model load until you pass an explicit model "
            "path; they will not report a verdict they did not reach.")
    endif()
endif()

option(HERD_BUILD_MODULES "Build the self-developed libraries and drivers" ON)
if(HERD_BUILD_MODULES)
    add_subdirectory(modules)

    # NOTE: a guarded link from `herd_seams` to `tq_reader_graph` lived here. It existed for exactly
    # one caller -- the attention replacement in `patches/herd_reader_attn.cpp` -- and that caller is
    # deleted (see the note at the probe block there; the reasons were a per-layer throttle that
    # produced a mixed result, a use after move, and an emission whose gather never existed). With the
    # call gone, both the link and its include directory are removed: an unused link compiles a
    # `-Werror` module into every product build and makes that build fail on code it never runs.
    #
    # If a future patch needs the emitter, the working shape is exactly what was deleted: put the
    # `if(TARGET tq_reader_graph)` block AFTER this `add_subdirectory`, never at `herd_seams`'s own
    # definition site (the target does not exist yet there, and `if(TARGET ...)` then silently skips
    # the link). `tq_reader_graph` lives outside `if(BUILD_TESTING)` and carries
    # `EXCLUDE_FROM_ALL`, so it is linkable from a product build and costs nothing until linked.
endif()

# The graph-owned index publisher, driven by upstream's real `llm_graph_result`.
#
# IT IS REGISTERED HERE, AFTER `add_subdirectory(modules)`, AND THAT IS NOT STYLE. `tq_reader_graph`
# is created inside `modules/reader/CMakeLists.txt`; an `if(TARGET tq_reader_graph)` at the patch
# tests' own site (around line 320) is evaluated before that subdirectory runs, so it is FALSE and the
# target is silently never created. That failure is invisible -- no error, no test, just an absence --
# and it is the second time this file has had to say so, which is why the note is here in the block
# rather than in a comment at the old site.
#
# It links `llama` because that is the only way to test the object: `llm_graph_result` is the class
# upstream's decode path uses to own graph inputs, and a stand-in could only ever restate the
# stand-in. It links `tq_reader_graph` because the publisher writes what the emitter produced; a
# second implementation of the tiling rules inside patches/ is exactly the duplication
# `publish_index_inputs` exists to prevent.
#
# The four cases it asserts are the four ways the input can be misused -- published before placement,
# published correctly, published twice, and read after teardown -- and each must be loud. The test
# fixture this replaces returned silently on the repeat, which is the failure mode a caller cannot
# detect.
if(TARGET llama AND TARGET tq_reader_graph)
    add_executable(herd-reader-publisher-test patches/herd_reader_publisher_test.cpp)
    target_include_directories(herd-reader-publisher-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/patches
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/reader/include)
    target_compile_options(herd-reader-publisher-test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
    target_link_libraries(herd-reader-publisher-test PRIVATE
        tq_reader_graph llama ggml ggml-cpu ggml-base)
    add_test(NAME herd.reader_publisher COMMAND herd-reader-publisher-test)

    # The PREFILL publisher: one graph-owned input that writes T per-token row tables in one pass.
    #
    # Registered in this same block, for the same reason as the target above -- `tq_reader_graph` does
    # not exist before `add_subdirectory(modules)` -- and linked the same way, because it writes what
    # `emit_prefill_attention` produced and must not grow a second copy of the tiling rules.
    #
    # Its refusing cases are observed as PROCESS DEATHS rather than caught exceptions: `set_input`
    # sits under `llama_process` (extern "C") with no `catch` on the chain, so the failure channel is
    # `std::abort`, and a test can only fork and inspect how the child died. `fork`/`execv`/`waitpid`
    # is used instead of `std::system` because a shell turns a signal death into exit status 128+n.
    add_executable(herd-reader-prefill-publisher-test patches/herd_reader_prefill_publisher_test.cpp)
    target_include_directories(herd-reader-prefill-publisher-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/patches
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/upstream/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/reader/include)
    target_compile_options(herd-reader-prefill-publisher-test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
    target_link_libraries(herd-reader-prefill-publisher-test PRIVATE
        tq_reader_graph llama ggml ggml-cpu ggml-base)
    add_test(NAME herd.reader_prefill_publisher COMMAND herd-reader-prefill-publisher-test)
endif()

# The row-table VALIDATOR, reached from `patches/herd_reader_attn.cpp`'s
# `HERD_READER_VALIDATE_ROWS` seam.
#
# Appended here, after `add_subdirectory(modules)`, for the reason stated twice already in this file:
# `tq_reader_rows` is created inside `modules/reader/CMakeLists.txt`, so an `if(TARGET ...)` placed
# next to `herd_seams`'s own definition is evaluated before the target exists and silently skips the
# link -- a failure with no error message and no test, just an absence.
#
# GUARDED because `tq_reader_rows` does not exist when modules are off. The patch TU handles that
# case itself: its `__has_include` check compiles a stub that names the missing validator and stops,
# so a build without it fails loudly when the switch is set instead of pretending to have verified.
if(TARGET tq_reader_rows)
    target_include_directories(herd_seams PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/reader/include)
    target_link_libraries(herd_seams PUBLIC tq_reader_rows)
    # The definition, not `__has_include`: it is set exactly when the library is linked, so the patch
    # TU cannot compile a stub that claims the switch is unsupported while the code is available.
    target_compile_definitions(herd_seams PRIVATE HERD_ROWS_VALIDATOR=1)
endif()

# The PHYSICAL emitter, reached from `patches/herd_reader_attn.cpp`'s `HERD_READER_GRAPH_K2` seam.
#
# Appended here for the same reason as the block above, and it is the third time this file records
# it: `herd_seams` is defined near the top, BEFORE `add_subdirectory(modules)`, so an `if(TARGET
# tq_reader)` written next to its definition is evaluated while `tq_reader` does not yet exist and
# silently skips the link. The result would be a patch TU that compiles a stub, a switch that does
# nothing, and no error anywhere.
#
# `tq_reader` is deliberately ggml-free (modules/reader/CMakeLists.txt), so linking it into the seam
# object library does not drag ggml into the reader -- the dependency runs one way only.
if(TARGET tq_reader_graph)
    target_include_directories(herd_seams PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/reader/include)
    # `tq_reader_graph`, NOT `tq_reader`: the emitter builds ggml nodes, so it lives in the
    # graph-facing library while `tq_reader` stays ggml-free by design. The first version of this
    # block linked `tq_reader` and produced five undefined references at link time -- the exact
    # failure mode a `TARGET` guard cannot catch, because the target it named does exist.
    target_link_libraries(herd_seams PUBLIC tq_reader_graph)
    # The definition, not `__has_include`: it is set exactly when the emitter is actually linked, so
    # the patch TU cannot compile a stub that claims K2 is unsupported while the code is right there.
    target_compile_definitions(herd_seams PRIVATE HERD_READER_PHYSICAL=1)
endif()

# ---------------------------------------------------------------------------------------------
# `herd.reader_request`, registered HERE because the question it needs answered is not answerable
# anywhere earlier.
#
# The test takes a binary as argv[1], an optional model as argv[2] and an optional corpus DIRECTORY
# as argv[3]. Cases 4-5 run `score -m MODEL --corpus DIR`, and that subcommand exists in exactly one
# binary of this tree: `llama-corpus`. Measured both ways rather than read off the usage line --
# `tq-batch-compose score …` answers `unknown option: score`, while `llama-corpus score …` loads the
# model and proceeds. (llama-corpus both builds a corpus and scores one; the test uses the second.)
#
# It lives in `modules/`, so `$<TARGET_FILE:...>` on it fails at GENERATE time when modules are off
# (`No target "llama-corpus"`), which turned a missing optional component into a tree that would not
# configure. The guard below is what answers that: registered after `add_subdirectory(modules)`, the
# target list is final, so `TARGET llama-corpus` is a real question -- and a build without modules
# skips the registration instead of failing. Guarding the same question earlier in this file would be
# silently FALSE even with modules on, which is the trap recorded twice above.
#
# The executable is built regardless; it links only the runtime and llama.
if(TARGET llama-corpus AND TARGET herd-reader-request-test)
    # The model falls back to the shared build-tree fixture. An empty default used to mean the two
    # attention-reaching cases silently did not run: the binary prints `skipped` for them and the
    # test still exits 0, so "no model" and "attention verified" were the same colour of green.
    # `TQ_TINY_LLAMA_MODEL` is readable here because this block runs after its definition.
    #
    # The CORPUS is a second, separate input (argv[3]) and it was NEVER supplied -- not by the old
    # `if(...)` registration and not by any default. Cases 4-5 need `score -m MODEL --corpus DIR`,
    # so in this tree they have never run: the test printed "cases 4-5 were NOT run" and exited 0.
    # There is no corpus directory in the source tree to point at, and generating one needs
    # `llama-corpus build --input <prompts>`, which is a fixture this registration does not own.
    # So the corpus stays an explicit operator input; absent it, CMake says so at configure time
    # instead of leaving a green row that verified only the constructor.
    set(_herd_reader_request_model "${HERD_READER_REQUEST_TEST_MODEL}")
    set(_herd_reader_request_uses_fixture FALSE)
    if(NOT _herd_reader_request_model)
        set(_herd_reader_request_model "${TQ_TINY_LLAMA_MODEL}")
        set(_herd_reader_request_uses_fixture TRUE)
    endif()
    set(HERD_READER_REQUEST_TEST_CORPUS "" CACHE PATH
        "Corpus DIRECTORY for herd.reader_request cases 4-5 (`score -m MODEL --corpus DIR`). \
Without it those two cases do not run; build one with `llama-corpus build -m MODEL --input DIR`.")
    if(HERD_READER_REQUEST_TEST_CORPUS)
        add_test(NAME herd.reader_request
                 COMMAND herd-reader-request-test $<TARGET_FILE:llama-corpus>
                         "${_herd_reader_request_model}"
                         "${HERD_READER_REQUEST_TEST_CORPUS}")
    else()
        add_test(NAME herd.reader_request
                 COMMAND herd-reader-request-test $<TARGET_FILE:llama-corpus>
                         "${_herd_reader_request_model}")
        message(STATUS
            "herd.reader_request: no HERD_READER_REQUEST_TEST_CORPUS, so cases 4-5 (the refusal "
            "at attention) will NOT run. This build contains no corpus directory and does not "
            "generate one; the test reports that gap on stderr and passes on the constructor "
            "checks alone.")
    endif()
    if(_herd_reader_request_uses_fixture AND TQ_TINY_LLAMA_FIXTURE)
        set_tests_properties(herd.reader_request PROPERTIES FIXTURES_REQUIRED tq_tiny_llama)
    endif()
endif()
