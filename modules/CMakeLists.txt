cmake_minimum_required(VERSION 3.21)

# Self-developed layer. Every source here is original project code; nothing in
# third_party/upstream is edited or copied. The C drivers consume only the
# public llama/ggml interfaces, exactly as they did against an installed
# package; here the same targets come from the in-tree submodule.
#
# Contract with the root project:
#   * target `llama` and `ggml::ggml-base` already exist, or
#     TQ_HERD_AUTO_UPSTREAM=ON lets this file add third_party/upstream itself.
#   * TQ_HERD_BUILD_APPS=ON adds the C drivers (needs the upstream targets).
#   * BUILD_TESTING gates the module tests through the parent project.

option(TQ_HERD_AUTO_UPSTREAM "Add third_party/upstream when llama is absent." OFF)
option(TQ_HERD_BUILD_APPS "Build the C driver programs." ON)
option(TQ_HERD_INSTALL_EXPORTS "Install CMake export sets for targets that link llama." OFF)

if(NOT TARGET llama AND TQ_HERD_AUTO_UPSTREAM)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../third_party/upstream"
                     "${CMAKE_CURRENT_BINARY_DIR}/upstream" EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()
# In-tree ggml exports plain targets; an installed package exports the
# ggml:: namespace. Normalize the namespace here, once.
foreach(_tq_pair "ggml::ggml-base;ggml-base" "ggml::ggml;ggml")
    list(GET _tq_pair 0 _alias)
    list(GET _tq_pair 1 _real)
    if(NOT TARGET ${_alias} AND TARGET ${_real})
        add_library(${_alias} ALIAS ${_real})
    endif()
endforeach()
if(NOT TARGET ggml::ggml-base)
    find_package(ggml CONFIG REQUIRED)
endif()

# imatrix-compose asks for the ggml include directory by name. An installed
# package exports it; in-tree it comes from the target's interface.
if(NOT GGML_INCLUDE_DIR)
    get_target_property(_tq_ggml_inc ggml::ggml-base INTERFACE_INCLUDE_DIRECTORIES)
    if(_tq_ggml_inc)
        list(GET _tq_ggml_inc 0 GGML_INCLUDE_DIR)
    endif()
endif()

# Libraries with no upstream dependency. Order is the dependency order.
add_subdirectory(sampling)
add_subdirectory(corpus)
add_subdirectory(scheduling)
add_subdirectory(text)
add_subdirectory(calibration)
add_subdirectory(relay)
add_subdirectory(rerot)
add_subdirectory(alloc-ledger)

# The TP5 device plane as a ggml backend PLUGIN (a shared object that upstream's own loader
# opens). Gated on HIP: this module's purpose is a real device, and a build without ROCm would
# be a stub that reports no device -- the CPU-simulation this project refuses to pass off as a
# GPU path. The module skips itself and says so.
add_subdirectory(relay-hip)
# Capabilities the fork declared in upstream's header, re-expressed as ours.
add_subdirectory(upstream-ext)
# The relay lane's policy (plan cache, frame identity, pool arithmetic, switch
# matrix). No ggml, no GPU: the decisions the relay backend needs, testable
# without one.
add_subdirectory(relay-plan)
# The relay control plane, executable on host memory: the same generation/ready/ack
# handshake the GPU transport speaks, with real threads, so its rules are measured
# rather than asserted.
add_subdirectory(relay-transport)
# Decoupled KV capacity: the legality and expressibility of a requested capacity,
# as arithmetic. No llama, no model -- the cache's size is chosen inside upstream
# from a const reference, so this module decides rather than allocates.
add_subdirectory(kv-capacity)
# The reader lane's rules (RERoT): view validation and resolution to spans. Pure
# domain core; the graph construction that consumes it is not reachable without an
# upstream struct change, which its README states.
add_subdirectory(reader)

# The RERoT whiteboard: the shared board a body step publishes to, as a station.
# AXIOM 1 (peers' last committed token), AXIOM 3 (no partial publish), LEMMA 2 (one
# step, one token per bound stage). Ledger only -- no llama, no ggml, no sequence
# handle: which stage is bound to which sequence stays the caller's business
# (LEMMA 3), so this compiles and its rules are measured without a model.
add_library(tq_rerot_whiteboard STATIC rerot-whiteboard/src/whiteboard.cpp)
add_library(TQRerot::whiteboard ALIAS tq_rerot_whiteboard)
set_target_properties(tq_rerot_whiteboard PROPERTIES EXPORT_NAME whiteboard)
target_compile_features(tq_rerot_whiteboard PUBLIC cxx_std_17)
target_compile_options(tq_rerot_whiteboard PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
target_include_directories(tq_rerot_whiteboard PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rerot-whiteboard/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(BUILD_TESTING)
    add_executable(tq_rerot_whiteboard_test rerot-whiteboard/tests/whiteboard-test.cpp)
    target_link_libraries(tq_rerot_whiteboard_test PRIVATE tq_rerot_whiteboard)
    target_compile_options(tq_rerot_whiteboard_test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
    add_test(NAME tq_rerot.whiteboard COMMAND tq_rerot_whiteboard_test)
endif()

if(TQ_HERD_BUILD_APPS)
    add_subdirectory(compose)
    add_subdirectory(corpus-compose)
    add_subdirectory(batch-compose)
    add_subdirectory(logits-compose)
    add_subdirectory(imatrix-compose)
    add_subdirectory(rerot-compose)
    # Entering tp5-compose is NOT the same decision as building its relay runtime, and
    # the two now have an option each. TQ_HERD_WANT_TP5 (this file) decides whether the
    # directory is entered; TQ_TP5_BUILD_RUNTIME (inside the package, default OFF)
    # decides whether `tq_tp5_runtime` and its driver are built. Entering therefore buys
    # the six relay-free tests -- ordered-fold, host-pool, mtp-state-checkpoint,
    # mtp-greedy-cache, relay-channel-map, commit-stop -- which compile against upstream
    # alone. Those six targets used to sit behind a `return()` that refused the whole
    # directory, so 9408 lines carried no regression net at all: a red build was not the
    # protection, it was the reason nothing was ever exercised.
    #
    # The runtime still refuses, with the exact missing pieces named (ggml-relay.h, the
    # HIP staging pool, relay.hip), because the relay is a backend and a backend is
    # registered rather than intercepted by a seam. Refusing beats shipping a target that
    # cannot work: a red target invites someone to "fix" it by stubbing the relay.
    #
    # Guarding at the parent as well as inside the package is deliberate: a directory that
    # is never entered cannot contribute a target by any path.
    # Declared here, before the guard that reads it. It used to live only inside
    # tp5-compose/CMakeLists.txt, which this guard decides whether to enter -- so on
    # a fresh tree `-DTQ_HERD_WANT_TP5=ON` was read as undefined, the subdirectory was
    # skipped, and the request was silently ignored: no target, no message, and a
    # cache entry that never appeared. RERoT does not have the bug because its
    # subdirectory is entered unconditionally, which is why its option shows up in
    # the cache. A request for an unavailable component has to produce the
    # explanation, not vanish.
    option(TQ_HERD_WANT_TP5 "Enter tp5-compose: builds its six relay-free tests (the relay runtime itself is a separate option, TQ_TP5_BUILD_RUNTIME, default OFF)" OFF)
    if(TQ_HERD_WANT_TP5)
        add_subdirectory(tp5-compose)
    endif()
    add_subdirectory(relay-probe)
endif()

# ---------------------------------------------------------------------------
# Attach the seam runtime to the real products.
#
# An OBJECT library's objects are added unconditionally to the consumer, so this
# carries the constructor and the `herd_seams` descriptors into the image. A
# build that links only `llama` gets no seams and cannot notice: there is no
# undefined symbol to fail on. Attaching here, once, for every product, is the
# difference between a patch layer and a test fixture.
#
# Deliberately not attached to unit tests: a test that asserts an upstream
# function's exact behaviour must see upstream, not a patched entry.
# ---------------------------------------------------------------------------
option(TQ_HERD_ATTACH_SEAMS "Install seams in the product binaries" ON)
if(TQ_HERD_ATTACH_SEAMS AND TARGET herd::seams)
    foreach(_tq_product
            llama-corpus tq_logits_compose tq_batch_compose
            tq-compose llama-compose tq_rerot_compose)
        if(TARGET ${_tq_product})
            target_link_libraries(${_tq_product} PRIVATE herd::seams)
        endif()
    endforeach()
endif()
