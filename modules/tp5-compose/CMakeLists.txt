cmake_minimum_required(VERSION 3.21)
project(TQTP5Compose VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# What cannot be built, and what can.
#
# The driver and its runtime are written against a relay backend the fork carried
# inside ggml: ggml-relay.h (event ABI, 5 ranks, plan identity), the HIP staging
# pool, and relay.hip itself. None of it exists in upstream, and none of it can be
# supplied by a function-entry seam: the relay is a *backend*, and a backend is
# registered, not intercepted. A working relay is a module of its own, verified on
# the five-GPU rig.
#
# The relay-free parts of this package are ordinary C++ and are NOT lost: the
# plan/frame/pool policy already moved to modules/relay-plan (built and tested), and
# ordered-fold, the MTP caches, the greedy cache, commit-stop, the host pool and the
# relay channel map compile against upstream alone. Leaving those targets unreachable
# until the relay exists meant 9408 lines carried no regression net -- so the gate
# below was split in two.
#
# ONE FACT, ONE OWNER. "Enter this directory" (TQ_HERD_WANT_TP5, declared by the
# parent) is not "build the relay runtime" (TQ_TP5_BUILD_RUNTIME, declared here).
# The first used to imply the second, which meant a request for the relay-free tests
# was answered with a FATAL_ERROR and no tests. Now the directory can be entered for
# the tests alone, and the runtime still refuses with the exact reason.
#
# Refusing beats shipping a target that cannot work: a red build invites someone to
# "fix" it by stubbing the relay, which is exactly the fake this project forbids.
# ---------------------------------------------------------------------------
option(TQ_TP5_BUILD_RUNTIME
    "Build tq_tp5_runtime and tq-tp5-compose (needs the relay backend, which does not exist)" OFF)
if(TQ_TP5_BUILD_RUNTIME)
    message(FATAL_ERROR
        "tp5-compose's runtime and driver cannot be built: src/runtime.cpp and src/main.cpp need "
        "ggml-relay.h (relay event ABI, 5 ranks, plan identity) and the HIP staging pool. "
        "The relay is a backend, not a function: implement it as a module registered through "
        "ggml_backend_register()/GGML_BACKEND_PATH over five devices, verified on the 5x GPU rig. "
        "The pure policy is already migrated to modules/relay-plan; the remaining relay-free "
        "sources (ordered-fold, MTP caches, host pool, channel map) compile against upstream and "
        "are built by the default configuration with -DTQ_HERD_WANT_TP5=ON. "
        "Drop -DTQ_TP5_BUILD_RUNTIME=ON to get the tests without the runtime.")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)
if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()
if(NOT TARGET TQRelay::relay)
    find_package(TQRelay CONFIG REQUIRED)
endif()
if(NOT TARGET TQSampling::sampling)
    find_package(TQSampling CONFIG REQUIRED)
endif()
if(NOT TARGET TQScheduling::scheduling)
    find_package(TQScheduling CONFIG REQUIRED)
endif()
if(NOT TARGET TQText::text)
    find_package(TQText CONFIG REQUIRED)
endif()

if(TQ_TP5_BUILD_RUNTIME)
    add_library(tq_tp5_runtime STATIC src/runtime.cpp src/ordered-fold.cpp)
    add_library(TQTP5Compose::runtime ALIAS tq_tp5_runtime)
    set_target_properties(tq_tp5_runtime PROPERTIES EXPORT_NAME runtime)
    target_compile_features(tq_tp5_runtime PUBLIC cxx_std_17)
    target_compile_options(tq_tp5_runtime PRIVATE -fno-fast-math -ffp-contract=off)
    target_include_directories(tq_tp5_runtime PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    # In-memory lease records come from the read-only F2 ledger library. Only its
    # value types are used here; the C adapter owns the external schema.
    target_include_directories(tq_tp5_runtime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../alloc-ledger/include)
    target_link_libraries(tq_tp5_runtime PRIVATE llama TQRelay::relay
        TQSampling::sampling TQScheduling::scheduling TQText::text Threads::Threads
        ${CMAKE_DL_LIBS})

    add_executable(tq_tp5_compose src/main.cpp)
    set_target_properties(tq_tp5_compose PROPERTIES OUTPUT_NAME tq-tp5-compose)
    target_link_libraries(tq_tp5_compose PRIVATE TQTP5Compose::runtime)
    if(UNIX AND NOT APPLE)
        set_target_properties(tq_tp5_compose PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(TARGETS tq_tp5_runtime tq_tp5_compose EXPORT TQTP5ComposeTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    configure_package_config_file(cmake/TQTP5ComposeConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TQTP5Compose")
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
    # In-tree consumers link the uninstalled upstream target, so the
    # exported package is opt-in until upstream is installed alongside it.
    if(TQ_HERD_INSTALL_EXPORTS)
        install(EXPORT TQTP5ComposeTargets NAMESPACE TQTP5Compose::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TQTP5Compose")
    endif()
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TQTP5Compose")
    install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQTP5Compose")
    install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQTP5Compose")
    install(PROGRAMS tests/real-tp5.py DESTINATION "${CMAKE_INSTALL_DATADIR}/TQTP5Compose" RENAME real-tp5.py)
endif()

include(CTest)
option(TQ_TP5_BUILD_HIP_TESTS "Build and run the real Q8/F32 and attention HIP arithmetic oracles" OFF)
if(BUILD_TESTING)
    # The driver's --help smoke is the only test that needs the runtime, so it is
    # the only one under the gate.
    if(TQ_TP5_BUILD_RUNTIME)
        add_test(NAME tq-tp5-compose-help COMMAND tq_tp5_compose --help)
    endif()

    # ---- relay-free from here down: upstream, sampling and text only. --------
    add_executable(tq_tp5_ordered_fold_tests tests/test-ordered-fold.cpp src/ordered-fold.cpp)
    target_compile_features(tq_tp5_ordered_fold_tests PRIVATE cxx_std_17)
    target_compile_options(tq_tp5_ordered_fold_tests PRIVATE -fno-fast-math -ffp-contract=off)
    target_include_directories(tq_tp5_ordered_fold_tests PRIVATE src)
    target_link_libraries(tq_tp5_ordered_fold_tests PRIVATE llama)
    add_test(NAME tq-tp5-ordered-f32-fold COMMAND tq_tp5_ordered_fold_tests)
    if(UNIX AND NOT APPLE)
        add_library(tp5_fail_malloc SHARED tests/fail-malloc.c)
        target_link_libraries(tp5_fail_malloc PRIVATE ${CMAKE_DL_LIBS})
    endif()
    add_executable(tq_tp5_host_pool_tests tests/test-host-pool.cpp)
    target_compile_features(tq_tp5_host_pool_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_host_pool_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../alloc-ledger/include)
    target_compile_definitions(tq_tp5_host_pool_tests PRIVATE
        TP5_TEST_VOCAB_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../../models/ggml-vocab-command-r.gguf")
    target_link_libraries(tq_tp5_host_pool_tests PRIVATE llama ${CMAKE_DL_LIBS})
    # `LD_PRELOAD` puts our malloc interposer ahead of libasan, so libasan loses
    # the first slot it requires and aborts before main. malloc interposition and
    # ASan's own interposition are mutually exclusive by construction, so the
    # loaded variant is registered only on a non-sanitized build; the plain
    # variant below always runs. The fail-malloc negative path is therefore
    # unexercised under ASan, not silently skipped.
    if(GGML_SANITIZE_ADDRESS OR TQ_ALLOC_SANITIZE)
        set(TP5_FAIL_MALLOC_TEST_AVAILABLE OFF)
    else()
        set(TP5_FAIL_MALLOC_TEST_AVAILABLE ON)
    endif()
    if(UNIX AND NOT APPLE AND TP5_FAIL_MALLOC_TEST_AVAILABLE)
        # Wrap with cmake -E env: ENVIRONMENT genex expansion is not
        # guaranteed at the declared CMake 3.21 minimum.
        add_test(NAME tq-tp5-host-pool COMMAND ${CMAKE_COMMAND} -E env
            "LD_PRELOAD=$<TARGET_FILE:tp5_fail_malloc>"
            "TP5_REQUIRE_FAIL_MALLOC=1"
            $<TARGET_FILE:tq_tp5_host_pool_tests>)
    else()
        add_test(NAME tq-tp5-host-pool COMMAND tq_tp5_host_pool_tests)
    endif()
    add_executable(tq_tp5_mtp_checkpoint_tests tests/test-mtp-state-checkpoint.cpp)
    target_compile_features(tq_tp5_mtp_checkpoint_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_mtp_checkpoint_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME tq-tp5-mtp-state-checkpoint COMMAND tq_tp5_mtp_checkpoint_tests)

    # tests/test-mtp-state-real.cpp includes upstream's private ggml-backend-impl.h
    # and is the only other target that needs the driver's context, so it stays
    # under the same gate rather than being repaired into the relay-free set.
    if(TQ_TP5_BUILD_RUNTIME)
        add_executable(tq_tp5_mtp_state_real_tests tests/test-mtp-state-real.cpp)
        target_compile_features(tq_tp5_mtp_state_real_tests PRIVATE cxx_std_17)
        target_link_libraries(tq_tp5_mtp_state_real_tests PRIVATE llama)
        add_test(NAME tq-tp5-mtp-state-real COMMAND tq_tp5_mtp_state_real_tests)
        set_tests_properties(tq-tp5-mtp-state-real PROPERTIES SKIP_RETURN_CODE 77)
        # The model comes from the operator: there is no generator for a Mamba-architecture GGUF in
        # this tree. This block used to name `tests/rsz/make_tiny_rsz_mamba.py` behind an `EXISTS`
        # guard; the file is not on disk, so the guard was always false, no fixture rule was ever
        # registered, and the test skipped for a reason nobody could see from the build output.
        # Writing a Mamba GGUF here needs a writer this tree does not have (no torch, no gguf
        # package), so the honest configuration is an explicit opt-in path and a visible skip.
        set(TP5_TEST_MODEL "" CACHE FILEPATH
            "Mamba-architecture GGUF for tq-tp5-mtp-state-real. No generator ships with this tree, \
so the test skips (return code 77) until one is provided.")
        if(TP5_TEST_MODEL)
            set_tests_properties(tq-tp5-mtp-state-real PROPERTIES
                ENVIRONMENT "TP5_TEST_MODEL=${TP5_TEST_MODEL}")
        else()
            message(STATUS
                "tq-tp5-mtp-state-real: no TP5_TEST_MODEL; the test will SKIP (77). A Mamba fixture "
                "cannot be generated here -- tests/rsz/make_tiny_rsz_mamba.py does not exist and "
                "this tree has no GGUF writer for that architecture.")
        endif()
    endif()

    add_executable(tq_tp5_mtp_greedy_cache_tests tests/test-mtp-greedy-cache.cpp)
    target_compile_features(tq_tp5_mtp_greedy_cache_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_mtp_greedy_cache_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tq_tp5_mtp_greedy_cache_tests PRIVATE TQSampling::sampling)
    add_test(NAME tq-tp5-mtp-greedy-cache COMMAND tq_tp5_mtp_greedy_cache_tests)
    add_executable(tq_tp5_relay_channel_map_tests tests/test-relay-channel-map.cpp)
    target_compile_features(tq_tp5_relay_channel_map_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_relay_channel_map_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tq_tp5_relay_channel_map_tests PRIVATE TQRelay::relay)
    add_test(NAME tq-tp5-relay-channel-map COMMAND tq_tp5_relay_channel_map_tests)
    add_executable(tq_tp5_commit_stop_tests tests/test-commit-stop.cpp)
    target_compile_features(tq_tp5_commit_stop_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_commit_stop_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tq_tp5_commit_stop_tests PRIVATE TQText::text)
    add_test(NAME tq-tp5-commit-stop COMMAND tq_tp5_commit_stop_tests)
    if(TQ_TP5_BUILD_HIP_TESTS)
        add_executable(tq_tp5_q8_f32_tests tests/test-q8-f32.cpp)
        target_compile_features(tq_tp5_q8_f32_tests PRIVATE cxx_std_17)
        target_link_libraries(tq_tp5_q8_f32_tests PRIVATE llama)
        add_test(NAME tq-tp5-q8-f32-oracle COMMAND tq_tp5_q8_f32_tests)
        set_tests_properties(tq-tp5-q8-f32-oracle PROPERTIES TIMEOUT 120)
        add_executable(tq_tp5_flash_f32_tests tests/test-flash-f32.cpp)
        target_compile_features(tq_tp5_flash_f32_tests PRIVATE cxx_std_17)
        target_compile_options(tq_tp5_flash_f32_tests PRIVATE -fno-fast-math -ffp-contract=off)
        target_link_libraries(tq_tp5_flash_f32_tests PRIVATE llama)
        add_test(NAME tq-tp5-flash-f32-oracle COMMAND tq_tp5_flash_f32_tests)
        set_tests_properties(tq-tp5-flash-f32-oracle PROPERTIES TIMEOUT 120)
    endif()
endif()
