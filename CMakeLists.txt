cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkVisualizer VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(MSVC)
    add_compile_options(/utf-8)
    add_compile_options(/W3)
endif()

find_package(Qt6 COMPONENTS Widgets QUIET)
if(Qt6_FOUND)
    message(STATUS "Using Qt6")
    set(QT_VERSION_MAJOR 6)
else()
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    message(STATUS "Using Qt5")
    set(QT_VERSION_MAJOR 5)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(project_thread_deps INTERFACE)
target_link_libraries(project_thread_deps INTERFACE Threads::Threads)

if(MINGW)
    target_link_libraries(project_thread_deps INTERFACE winpthread)
endif()

set(PROJECT_SOURCES
    src/main.cpp
    src/neural_network.cpp
    src/mainwindow.cpp
    src/network_view.cpp
    src/loss_chart.cpp
    src/training_thread.cpp
    src/cnn/tensor.cpp
    src/cnn/random.cpp
    src/cnn/conv_layer.cpp
    src/cnn/pooling_layer.cpp
    src/cnn/flatten_layer.cpp
    src/cnn/cnn_network.cpp
    src/cnn/cnn_training_thread.cpp
    src/cnn_mainwindow.cpp
    src/visualization/cnn_view.cpp
    src/visualization/feature_map_view.cpp
)

set(PROJECT_HEADERS
    include/neural_network.h
    include/mainwindow.h
    include/network_view.h
    include/loss_chart.h
    include/training_thread.h
    include/cnn/tensor.h
    include/cnn/cnn_layer_base.h
    include/cnn/conv_layer.h
    include/cnn/pooling_layer.h
    include/cnn/flatten_layer.h
    include/cnn/cnn_network.h
    include/cnn/cnn_training_thread.h
    include/cnn_mainwindow.h
    include/visualization/cnn_view.h
    include/visualization/feature_map_view.h
)

add_executable(${PROJECT_NAME} WIN32
    ${PROJECT_SOURCES} 
    ${PROJECT_HEADERS}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

if(QT_VERSION_MAJOR EQUAL 6)
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Widgets)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE project_thread_deps)

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WIN32_EXECUTABLE ON
    )
endif()

add_subdirectory(tests)
