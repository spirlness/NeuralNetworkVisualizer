cmake_minimum_required(VERSION 3.16)

# Option to build tests (inherited from parent)
option(BUILD_TESTS "Build test executables" ON)
if(NOT BUILD_TESTS)
    return()
endif()

# Common source files for tests (no Qt dependencies)
set(TEST_COMMON_SOURCES
    ../src/neural_network.cpp
    ../src/cnn/random.cpp
    ../src/cnn/tensor.cpp
    ../src/cnn/conv_layer.cpp
    ../src/cnn/pooling_layer.cpp
    ../src/cnn/flatten_layer.cpp
    ../src/cnn/cnn_network.cpp
)

# Functional Test
add_executable(FunctionalTest
    functional_test.cpp
    ${TEST_COMMON_SOURCES}
)

target_include_directories(FunctionalTest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(FunctionalTest PRIVATE project_thread_deps)

# CNN Diagnostic Test
add_executable(CNNDiagnostic
    cnn_diagnostic.cpp
    ${TEST_COMMON_SOURCES}
)

target_include_directories(CNNDiagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(CNNDiagnostic PRIVATE project_thread_deps)

# CTest integration
enable_testing()
add_test(NAME FunctionalTest COMMAND FunctionalTest)
add_test(NAME CNNDiagnostic COMMAND CNNDiagnostic)

# Install targets
install(TARGETS FunctionalTest CNNDiagnostic
    RUNTIME DESTINATION bin/tests
)
