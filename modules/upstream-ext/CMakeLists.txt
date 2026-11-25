cmake_minimum_required(VERSION 3.16)
project(TQUpstreamExt VERSION 0.1.0 LANGUAGES CXX)

# Capabilities the fork declared inside upstream's header, re-expressed as ours.
# No llama link: every entry point takes an opaque llama_context pointer, and a
# patch module supplies the implementation at runtime.
add_library(tq_upstream_ext src/upstream-ext.cpp)
add_library(TQUpstreamExt::ext ALIAS tq_upstream_ext)
target_compile_features(tq_upstream_ext PUBLIC cxx_std_17)
target_include_directories(tq_upstream_ext PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(CTest)
if(BUILD_TESTING)
    add_executable(tq_upstream_ext_unavailable_test tests/test-unavailable.cpp)
    target_link_libraries(tq_upstream_ext_unavailable_test PRIVATE tq_upstream_ext)
    add_test(NAME tq_upstream_ext.unavailable COMMAND tq_upstream_ext_unavailable_test)

    # The registered-provider path: `captured` in the JSON, the context a provider is asked
    # about, and the refusal of a second registration. `tq_upstream_ext.unavailable` covers the
    # opposite end -- nothing registered -- so between the two the ABI has a test at each end
    # rather than only where it refuses.
    add_executable(tq_upstream_ext_prefab_provider_test tests/test-prefab-provider.cpp)
    target_link_libraries(tq_upstream_ext_prefab_provider_test PRIVATE tq_upstream_ext)
    add_test(NAME tq_upstream_ext.prefab_provider
             COMMAND tq_upstream_ext_prefab_provider_test)
endif()
