cmake_minimum_required(VERSION 3.16)
project(TQKvCapacity VERSION 0.1.0 LANGUAGES CXX)

# Decoupled KV capacity, as a decision. No llama, no ggml, no model: the arithmetic
# that says what a requested configuration means, whether it is legal, and whether
# upstream can produce it at all.
add_library(tq_kv_capacity src/capacity.cpp)
add_library(TQKvCapacity::capacity ALIAS tq_kv_capacity)
target_compile_features(tq_kv_capacity PUBLIC cxx_std_17)
target_compile_options(tq_kv_capacity PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
target_include_directories(tq_kv_capacity PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(CTest)
if(BUILD_TESTING)
    add_executable(tq_kv_capacity_tests tests/capacity-test.cpp)
    target_link_libraries(tq_kv_capacity_tests PRIVATE tq_kv_capacity)
    add_test(NAME tq_kv_capacity.contract COMMAND tq_kv_capacity_tests)
endif()
