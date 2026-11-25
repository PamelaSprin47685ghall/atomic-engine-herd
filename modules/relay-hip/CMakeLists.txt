cmake_minimum_required(VERSION 3.16)
project(TQRelayHip VERSION 0.1.0 LANGUAGES CXX)

# The TP5 device plane as a ggml backend PLUGIN.
#
# Not a library the host links, and not a patched translation unit: a shared object that
# upstream's own loader opens with ggml_backend_load(). The three properties that matter:
#
#   SHARED            the loader works on files, not on archives
#   no libggml link    the plugin talks to ggml only through the struct layouts in
#                      ggml-backend-impl.h and the symbols its host provides at load time.
#                      Linking libggml would be wrong as well as unnecessary: two copies of
#                      ggml in one process is how a backend ends up with a private registry
#                      that nobody can see.
#   GGML_BACKEND_DL    makes GGML_BACKEND_DL_IMPL emit ggml_backend_init / ggml_backend_score
#                      with the visibility upstream's dlsym looks for. Without it the macros
#                      expand to nothing and the plugin loads but exports nothing -- a failure
#                      that is silent at build time and confusing at load time.
#
# HIP is a hard requirement here, not an option: this module's purpose is a real device plane,
# and a build without HIP would be a stub that reports no device. A host without ROCm cannot
# configure this module, and that is the correct outcome -- better than a plugin that builds
# everywhere and means nothing.

find_package(hip QUIET)
if(NOT hip_FOUND)
    message(STATUS "TQRelayHip: HIP not found, module skipped (a TP5 plugin with no device backend would be a stub)")
    return()
endif()

set(TQ_HERD_GGML_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/upstream/ggml/src)

add_library(tq_relay_hip SHARED src/hip_backend.cpp)
add_library(TQRelayHip::plugin ALIAS tq_relay_hip)

# The plugin's public header (the probe and any host-side consumer include it), plus ggml's
# backend implementation header -- the one every backend in the tree implements against.
target_include_directories(tq_relay_hip PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(tq_relay_hip PRIVATE
    ${TQ_HERD_GGML_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/upstream/ggml/include)

target_compile_features(tq_relay_hip PRIVATE cxx_std_17)
target_compile_definitions(tq_relay_hip PRIVATE GGML_BACKEND_DL=1)
target_compile_options(tq_relay_hip PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)

# HIP headers and the runtime. `hip::host` is the C++ runtime target; the kernel language is
# not needed yet because there are no kernels -- this file is host code that drives the device.
target_link_libraries(tq_relay_hip PRIVATE hip::host)

set_target_properties(tq_relay_hip PROPERTIES
    OUTPUT_NAME "ggml-tp5"
    # Upstream's loader looks for libggml-<name>.so on Linux and ggml-<name>.dll on Windows;
    # the prefix/suffix default per platform already match that, so only the name is set.
    PREFIX "lib"
)

# The probe: a host that loads the plugin the way llama.cpp would, through ggml_backend_load.
# It links ggml itself, because being the host is what it is testing.
if(BUILD_TESTING AND TARGET llama)
    add_executable(tq-relay-hip-probe tests/plugin-probe.cpp)
    target_link_libraries(tq-relay-hip-probe PRIVATE llama tq_relay_hip)
    target_compile_options(tq-relay-hip-probe PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
    add_dependencies(tq-relay-hip-probe tq_relay_hip)
    # Not a ctest: it needs a GPU, and a test that skips itself on a machine without one is
    # indistinguishable from a test that passed. Run it by hand; the command is the evidence.
    add_test(NAME tq_relay_hip.plugin
             COMMAND tq-relay-hip-probe $<TARGET_FILE:tq_relay_hip>)
endif()

# The admission rule, on machines this host cannot become (no device, absent-overrides-count).
#
# It links the plugin, not the probe: the facts it passes are plain structs, so no GPU is
# needed for the score rows. The zero-byte round-trip row does call HIP, and refuses before
# touching the device -- so this test is safe on a host without ROCm too, and it says so
# rather than skipping silently.
add_executable(tq-relay-hip-admission-test tests/admission-test.cpp)
target_link_libraries(tq-relay-hip-admission-test PRIVATE tq_relay_hip)
target_compile_options(tq-relay-hip-admission-test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_relay_hip.admission COMMAND tq-relay-hip-admission-test)
