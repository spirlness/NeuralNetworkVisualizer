cmake_minimum_required(VERSION 3.16)

# 添加功能测试可执行文件
add_executable(FunctionalTest
    functional_test.cpp
    ../src/neural_network.cpp
    ../src/cnn/random.cpp
    ../src/cnn/tensor.cpp
    ../src/cnn/conv_layer.cpp
    ../src/cnn/pooling_layer.cpp
    ../src/cnn/flatten_layer.cpp
    ../src/cnn/cnn_network.cpp
)

target_include_directories(FunctionalTest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(FunctionalTest PRIVATE project_thread_deps)

add_executable(CNNDiagnostic
    cnn_diagnostic.cpp
    ../src/neural_network.cpp
    ../src/cnn/random.cpp
    ../src/cnn/tensor.cpp
    ../src/cnn/conv_layer.cpp
    ../src/cnn/pooling_layer.cpp
    ../src/cnn/flatten_layer.cpp
    ../src/cnn/cnn_network.cpp
)

target_include_directories(CNNDiagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(CNNDiagnostic PRIVATE project_thread_deps)

enable_testing()
add_test(NAME FunctionalTest COMMAND FunctionalTest)
