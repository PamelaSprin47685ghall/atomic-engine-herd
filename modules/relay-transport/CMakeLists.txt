cmake_minimum_required(VERSION 3.16)
project(TQRelayTransport VERSION 0.1.0 LANGUAGES CXX)

# The relay control plane and its host-memory transport. No ggml, no llama, no GPU:
# the plane is a fixed layout plus a release/acquire discipline, and both are
# executable on ordinary memory with ordinary threads.
add_library(tq_relay_transport STATIC src/loopback.cpp)
add_library(TQRelayTransport::transport ALIAS tq_relay_transport)
target_compile_features(tq_relay_transport PUBLIC cxx_std_17)
target_compile_options(tq_relay_transport PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
target_include_directories(tq_relay_transport PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
find_package(Threads REQUIRED)
target_link_libraries(tq_relay_transport PUBLIC Threads::Threads)

include(CTest)
if(BUILD_TESTING)
    add_executable(tq_relay_transport_tests tests/transport-test.cpp)
    target_link_libraries(tq_relay_transport_tests PRIVATE tq_relay_transport)
    add_test(NAME tq_relay_transport.protocol COMMAND tq_relay_transport_tests)
endif()
