cmake_minimum_required(VERSION 3.21)
project(TQRelayProbe VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

# Empty means "derive it": the tree should not ship a default that names one machine's card. The
# old default was a literal `gfx1030`, so on this box (gfx1103) the probe compiled fine and its
# binaries carried no code object for the GPU they would run on -- a green build that says nothing
# about the hardware in front of it.
#
# Order: the operator's own value, then the arch CMake is already targeting, then whatever
# `rocm_agent_enumerator` reports, and only then the historical literal as a last resort.
set(TQ_RELAY_PROBE_ARCHS "" CACHE STRING
    "HIP architectures for the probe. Empty derives from CMAKE_HIP_ARCHITECTURES, then from the local GPU.")
if(TQ_RELAY_PROBE_ARCHS)
    set(TQ_RELAY_PROBE_ARCHS_EFFECTIVE "${TQ_RELAY_PROBE_ARCHS}")
elseif(CMAKE_HIP_ARCHITECTURES)
    set(TQ_RELAY_PROBE_ARCHS_EFFECTIVE "${CMAKE_HIP_ARCHITECTURES}")
else()
    find_program(TQ_ROCM_AGENT_ENUMERATOR NAMES rocm_agent_enumerator
                 HINTS /opt/rocm/bin /opt/rocm/llvm/bin)
    if(TQ_ROCM_AGENT_ENUMERATOR)
        execute_process(COMMAND "${TQ_ROCM_AGENT_ENUMERATOR}" OUTPUT_VARIABLE _tq_agents
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REGEX MATCHALL "gfx[0-9a-z]+" _tq_agents "${_tq_agents}")
        if(_tq_agents)
            list(REMOVE_DUPLICATES _tq_agents)
            string(REPLACE ";" " " _tq_agents "${_tq_agents}")
            set(TQ_RELAY_PROBE_ARCHS_EFFECTIVE "${_tq_agents}")
        endif()
    endif()
    if(NOT TQ_RELAY_PROBE_ARCHS_EFFECTIVE)
        # Historical value, kept only so a HIP-less configuration still has something to name.
        # It is NOT this machine's arch; pass -DTQ_RELAY_PROBE_ARCHS to be sure.
        set(TQ_RELAY_PROBE_ARCHS_EFFECTIVE "gfx1030")
        message(STATUS
            "tq-relay-probe: no HIP arch could be derived; falling back to gfx1030. If your GPU is "
            "something else, pass -DTQ_RELAY_PROBE_ARCHS=<gfxNNNN> -- a probe built for another arch "
            "runs, but not on the kernels you meant to exercise.")
    endif()
endif()
# The operator's value wins over the derivation chain, and a CACHE entry survives from an earlier
# configuration on another machine. So a stale `gfx1030` in the cache compiles cleanly, prints the
# derivation chain as if it had been consulted, and ships code objects for a GPU that is not in the
# box. Ask the driver what is actually here and name any architecture it does not report.
find_program(TQ_ROCM_AGENT_ENUMERATOR NAMES rocm_agent_enumerator
             HINTS /opt/rocm/bin /opt/rocm/llvm/bin)
if(TQ_ROCM_AGENT_ENUMERATOR)
    execute_process(COMMAND "${TQ_ROCM_AGENT_ENUMERATOR}" OUTPUT_VARIABLE _tq_local_agents
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REGEX MATCHALL "gfx[0-9a-z]+" _tq_local_agents "${_tq_local_agents}")
    if(_tq_local_agents)
        list(REMOVE_DUPLICATES _tq_local_agents)
        set(TQ_RELAY_PROBE_LOCAL_ARCHS "${_tq_local_agents}")
    endif()
endif()

if(TQ_RELAY_PROBE_LOCAL_ARCHS)
    # The effective value is space-separated (it is passed to HIP_ARCHITECTURES as one string).
    string(REPLACE " " ";" _tq_effective_list "${TQ_RELAY_PROBE_ARCHS_EFFECTIVE}")
    set(_tq_unmatched "")
    foreach(_tq_arch ${_tq_effective_list})
        if(NOT _tq_arch IN_LIST TQ_RELAY_PROBE_LOCAL_ARCHS)
            list(APPEND _tq_unmatched "${_tq_arch}")
        endif()
    endforeach()
    if(_tq_unmatched)
        message(WARNING
            "tq-relay-probe: architecture(s) '${_tq_unmatched}' are not reported by "
            "rocm_agent_enumerator on this machine (local: ${TQ_RELAY_PROBE_LOCAL_ARCHS}). The probe "
            "would build code objects for a GPU that is not here. Clear a stale cache entry with "
            "-U TQ_RELAY_PROBE_ARCHS, or pass -DTQ_RELAY_PROBE_ARCHS=<gfxNNNN>.")
    endif()
endif()
message(STATUS "tq-relay-probe: HIP architectures = ${TQ_RELAY_PROBE_ARCHS_EFFECTIVE}")
if(TQ_RELAY_PROBE_LOCAL_ARCHS)
    message(STATUS "tq-relay-probe: local GPU architectures = ${TQ_RELAY_PROBE_LOCAL_ARCHS}")
endif()
option(TQ_RELAY_PROBE_BUILD_TESTS "Build the probe self-checks that need no GPU." ON)
option(TQ_RELAY_PROBE_BUILD_HIP "Build HIP tools when the ROCm toolchain is available." ON)

add_library(tq_relay_probe_common STATIC src/evidence.cpp)
target_compile_features(tq_relay_probe_common PUBLIC cxx_std_17)
target_include_directories(tq_relay_probe_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tq-relay-probe>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq_relay_probe_common PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(TQ_RELAY_PROBE_BUILD_TESTS)
    include(CTest)
    add_executable(tq_relay_probe_config_tests tests/config-test.cpp)
    target_link_libraries(tq_relay_probe_config_tests PRIVATE tq_relay_probe_common)
    add_test(NAME tq_relay_probe.config COMMAND tq_relay_probe_config_tests)
    # The evidence validator is the operator's pre-ship check; its self-test
    # runs here too so the pipeline proves the tool before anyone relies on
    # it. Python is already a fork-tooling dependency (build.py, guards).
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME tq_relay_probe.validator
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/validate-report.py --self-test)
endif()

if(NOT TQ_RELAY_PROBE_BUILD_HIP)
    message(STATUS "HIP tools disabled; building CPU evidence checks only")
    return()
endif()
find_package(hip QUIET)
if(NOT hip_FOUND)
    message(STATUS "HIP not found; the relay platform probe executable is not built")
    return()
endif()
enable_language(HIP)

# Native HIP compilation supplies the device link flags. Link the host
# runtime as a real imported library as well. Its directory can be an
# implicit HIP compiler search path, which CMake omits from automatic
# RPATH generation even though the OS loader does not know that path.
set(tq_relay_probe_hip_runtime_dir "$<TARGET_FILE_DIR:hip::amdhip64>")

# This module mixes two toolchains inside one image: `src/main.cpp` and `tools/tq-clock-rate.cpp`
# are compiled as HIP by ROCm's clang (its own ASan ABI, libclang_rt.asan / the ABI version tagged
# into `__asan_version_mismatch_check_v8`), while `tq_relay_probe_common` is compiled as C++ by the
# host compiler (GCC here, whose ASan ABI is /usr/lib/libasan.so.8). No link flag reconciles the two
# ABIs: the image aborts at exec with "Your application is linked against incompatible ASan
# runtimes." -- measured, unchanged by DT_NEEDED ordering, by `-static-libasan`, and by
# `-static-libsan`. Unifying the toolchain for this one module is a bigger decision than a link
# option, so the sanitizer build states the limit instead of faking a cure: the HIP executables are
# still built and still instrumented, and the Python CLI test that execs them is registered only
# when no sanitizer is active.
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=address" OR CMAKE_C_FLAGS MATCHES "-fsanitize=address"
   OR CMAKE_EXE_LINKER_FLAGS MATCHES "-fsanitize=address"
   OR GGML_SANITIZE_ADDRESS OR LLAMA_SANITIZE_ADDRESS OR TQ_ALLOC_SANITIZE)
    set(_tq_relay_probe_mixed_toolchain_asan ON)
else()
    set(_tq_relay_probe_mixed_toolchain_asan OFF)
endif()

add_executable(tq-relay-probe src/main.cpp)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(tq-relay-probe PRIVATE tq_relay_probe_common hip::host)
set_target_properties(tq-relay-probe PROPERTIES
    HIP_ARCHITECTURES "${TQ_RELAY_PROBE_ARCHS_EFFECTIVE}"
    BUILD_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
target_compile_features(tq-relay-probe PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq-relay-probe PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Clock-rate calibration utility for the operator: re-derives the clock64()
# rate the probe's spin budgets are anchored to, on the machine it will
# actually run on. Diagnostic only; produces no evidence files.
add_executable(tq-clock-rate tools/tq-clock-rate.cpp)
set_source_files_properties(tools/tq-clock-rate.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(tq-clock-rate PRIVATE hip::host)
set_target_properties(tq-clock-rate PROPERTIES
    HIP_ARCHITECTURES "${TQ_RELAY_PROBE_ARCHS_EFFECTIVE}"
    BUILD_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
target_compile_features(tq-clock-rate PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq-clock-rate PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TQ_RELAY_PROBE_BUILD_TESTS)
    # `tq-relay-probe` mixes two toolchains inside one image (HIP objects from ROCm's clang, C++
    # objects from the host compiler), so the two ASan ABIs cannot coexist: the binary aborts at
    # exec with "Your application is linked against incompatible ASan runtimes." That is a
    # measured property of this module, not a defect the CLI test can see past, so the test that
    # execs the binary is registered only when no sanitizer is active. The remaining relay-probe
    # tests (config, validator, offload_arch) still run under ASan.
    if(NOT _tq_relay_probe_mixed_toolchain_asan)
        add_test(NAME tq_relay_probe.cli
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.py
                         $<TARGET_FILE:tq-relay-probe> $<TARGET_FILE:tq-clock-rate>)
    endif()
    # A green build proves the toolchain ran; it does not prove the fatbin carries a code object
    # for the GPU in front of it. Assert that fact, so a stale cached architecture fails here
    # instead of passing as a successful probe on hardware it cannot execute on.
    if(TQ_RELAY_PROBE_LOCAL_ARCHS)
        find_program(TQ_RELAY_PROBE_LLVM_OBJDUMP NAMES llvm-objdump
                     HINTS /opt/rocm/llvm/bin /opt/rocm/bin)
        if(TQ_RELAY_PROBE_LLVM_OBJDUMP)
            add_test(NAME tq_relay_probe.offload_arch
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_offload_arch.py
                             ${TQ_RELAY_PROBE_LLVM_OBJDUMP}
                             $<TARGET_FILE:tq-relay-probe>
                             ${TQ_RELAY_PROBE_LOCAL_ARCHS})
        endif()
    endif()
endif()

install(TARGETS tq-relay-probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(TARGETS tq-clock-rate RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
# The evidence validator is part of the operator's pre-ship contract
# (README: archive evidence only after it validates); ship it with the
# binaries so an installed tree can stand alone. It is already executable
# in the repository; PROGRAMS preserves that mode.
install(PROGRAMS validate-report.py DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQRelayProbe")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQRelayProbe")
