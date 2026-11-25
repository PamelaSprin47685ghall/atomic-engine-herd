cmake_minimum_required(VERSION 3.16)
project(TQBatchCompose VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)
if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()
if(NOT TARGET TQScheduling::scheduling)
    find_package(TQScheduling CONFIG REQUIRED)
endif()
if(NOT TARGET TQText::text)
    find_package(TQText CONFIG REQUIRED)
endif()

add_executable(tq_batch_compose src/main.cpp)
set_target_properties(tq_batch_compose PROPERTIES OUTPUT_NAME tq-batch-compose)
if(UNIX AND NOT APPLE)
    set_target_properties(tq_batch_compose PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
target_compile_features(tq_batch_compose PRIVATE cxx_std_17)
target_link_libraries(tq_batch_compose PRIVATE llama TQScheduling::scheduling TQText::text herd_seams)
install(TARGETS tq_batch_compose RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQBatchCompose")

# The model comes from the shared build-tree fixture (`TQ_TINY_LLAMA_MODEL`, set by the top-level
# CMakeLists before `add_subdirectory(modules)`), not from /tmp. A /tmp wipe used to turn these
# three tests into failures at model load without touching anything this project owns.
set(TQ_BATCH_COMPOSE_TEST_MODEL "${TQ_TINY_LLAMA_MODEL}" CACHE FILEPATH
    "Real GGUF model for the integration test (defaults to the build-tree fixture).")
include(CTest)
if(BUILD_TESTING)
    add_test(NAME tq-batch-compose-help COMMAND tq_batch_compose --help)
    # No model needed: every assertion in this test happens before model loading,
    # which is the point -- an unavailable capability must be refused at the
    # argument, not discovered halfway through a run.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME tq_batch_compose.contract
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test-contract.py"
            --binary "$<TARGET_FILE:tq_batch_compose>"
            --model "")
endif()
set(TQ_BATCH_COMPOSE_TEST_GPU_LAYERS "0" CACHE STRING "GPU layers requested by the integration test.")
# Registered unconditionally: the model always has a home now. The old `if(...)` guard meant the
# default configuration silently contained no integration test at all.
add_test(NAME tq_batch_compose.real_model
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/real-batch.py"
        --binary "$<TARGET_FILE:tq_batch_compose>"
        --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
        --gpu-layers "${TQ_BATCH_COMPOSE_TEST_GPU_LAYERS}"
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/real-model-test")
add_test(NAME tq_batch_compose.admission
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_admission.py"
        --binary "$<TARGET_FILE:tq_batch_compose>"
        --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
        --gpu-layers "${TQ_BATCH_COMPOSE_TEST_GPU_LAYERS}"
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/admission-test")
add_test(NAME tq_batch_compose.streaming
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_streaming.py"
        --binary "$<TARGET_FILE:tq_batch_compose>"
        --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
        --gpu-layers "${TQ_BATCH_COMPOSE_TEST_GPU_LAYERS}"
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/streaming-test")
add_test(NAME tq_batch_compose.k2_parity
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_k2_parity.py"
        --binary "$<TARGET_FILE:tq_batch_compose>"
        --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/k2-parity-test")
add_test(NAME tq_batch_compose.k2c_prefill_parity
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_k2c_prefill_parity.py"
        --binary "$<TARGET_FILE:tq_batch_compose>"
        --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/k2c-prefill-parity-test")
if(TQ_TINY_LLAMA_FIXTURE AND TQ_BATCH_COMPOSE_TEST_MODEL STREQUAL TQ_TINY_LLAMA_MODEL)
    set_tests_properties(tq_batch_compose.real_model tq_batch_compose.admission
            tq_batch_compose.streaming tq_batch_compose.k2_parity
            tq_batch_compose.k2c_prefill_parity
        PROPERTIES FIXTURES_REQUIRED tq_tiny_llama)
endif()
