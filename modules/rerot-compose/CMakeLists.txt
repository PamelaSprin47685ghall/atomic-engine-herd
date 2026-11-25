cmake_minimum_required(VERSION 3.16)
project(TQRerotCompose VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()
if(NOT TARGET TQRerot::rerot)
    find_package(TQRerot CONFIG REQUIRED)
endif()
if(NOT TARGET TQRelay::relay)
    find_package(TQRelay CONFIG REQUIRED)
endif()
if(NOT TARGET TQText::text)
    find_package(TQText CONFIG REQUIRED)
endif()
if(NOT TARGET TQSampling::sampling)
    find_package(TQSampling CONFIG REQUIRED)
endif()

# tq-rerot-compose is deliberately not a target. RERoT is defined by a reader
# subsystem upstream does not have: reference attention over another sequence's
# stored K with a per-entry rotated Q, plus a KV capacity decoupled from the
# logical context. Neither is a function a seam can intercept -- they decide which
# keys attention reads and how much KV the cache allocates -- so there is no
# honest implementation for this driver to run.
#
# It used to build against a header that declared the fork's API with bodies that
# always returned false. That was a fake capability: the binary existed, started,
# and refused at run time, which reads like a disabled feature rather than an
# absent subsystem, and the answer was baked into the source so no provider could
# ever have flipped it.
#
# So the engine is kept unmodified as the specification of what the station must
# satisfy, it carries an #error, and the target is absent. Anyone who asks for it
# gets the requirement in one message instead of a binary that cannot work.
option(TQ_HERD_WANT_REROT "Build rerot-compose (needs the reader station, which does not exist)" OFF)
if(TQ_HERD_WANT_REROT)
    message(FATAL_ERROR
        "rerot-compose cannot be built: its engine needs the reader station -- "
        "llama_attention_{set,clear}_views over per-entry virtual_pos, and whiteboard capacity "
        "(n_rs_slots, n_ctx_kv, recurrent slot retire). Upstream has none of it, and no "
        "function-entry seam can decide which keys attention reads or how much KV the cache "
        "allocates. Implement the station, delete the #error in src/native.cpp, then restore "
        "add_executable(tq_rerot_compose src/main.cpp src/native.cpp) here.")
endif()
# The library-level tests do not touch llama_context, so they stay buildable and
# keep guarding the reusable parts of this package.
install(FILES README.md QUALIFICATION.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQRerotCompose")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQRerotCompose")

include(CTest)
if(BUILD_TESTING)
    add_executable(test_rerot_compose_sampling tests/test-sampling.cpp)
    target_compile_features(test_rerot_compose_sampling PRIVATE cxx_std_17)
    target_include_directories(test_rerot_compose_sampling PRIVATE src)
    target_link_libraries(test_rerot_compose_sampling PRIVATE llama TQSampling::sampling)
    add_test(NAME tq_rerot_compose.sampling COMMAND test_rerot_compose_sampling)
    add_executable(test_rerot_compose_capacity tests/test-capacity.cpp)
    target_compile_features(test_rerot_compose_capacity PRIVATE cxx_std_17)
    target_include_directories(test_rerot_compose_capacity PRIVATE src)
    add_test(NAME tq_rerot_compose.capacity COMMAND test_rerot_compose_capacity)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME tq_rerot_compose.whiteboard_validator
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test-whiteboard-validator.py")
    # Not registered: tq_rerot_compose.cancellation, .cli and .capabilities all
    # need the driver binary, which this build does not produce. They stay on disk
    # next to the engine so the station's implementation inherits ready-made
    # checks, but a test that cannot run must not appear as one that passed.
endif()
