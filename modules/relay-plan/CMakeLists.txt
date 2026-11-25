cmake_minimum_required(VERSION 3.16)
project(TQRelayPlan VERSION 0.1.0 LANGUAGES CXX)

# The relay lane's policy: plan identity and cache, frame identity, channel pool
# arithmetic, per-window channel ownership, and the switch matrix. No ggml, no
# llama, no GPU, no Dobby -- which is why it is testable here and the relay
# backend itself is not.
add_library(tq_relay_plan STATIC src/plan.cpp)
add_library(TQRelayPlan::plan ALIAS tq_relay_plan)
set_target_properties(tq_relay_plan PROPERTIES EXPORT_NAME plan)
target_compile_features(tq_relay_plan PUBLIC cxx_std_17)
target_compile_options(tq_relay_plan PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
target_include_directories(tq_relay_plan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(CTest)
if(BUILD_TESTING)
    # Distinct from modules/relay's tq_relay_plan_tests: that one tests the relay
    # transport library, this one the plan policy, and CMake target names are
    # global across the whole build.
    add_executable(tq_relay_plan_policy_tests tests/plan-test.cpp)
    target_link_libraries(tq_relay_plan_policy_tests PRIVATE tq_relay_plan)
    add_test(NAME tq_relay_plan.policy COMMAND tq_relay_plan_policy_tests)
endif()

include(GNUInstallDirs)
install(TARGETS tq_relay_plan ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
