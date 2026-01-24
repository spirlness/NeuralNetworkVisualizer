cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkVisualizer VERSION 2.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_GUI "Build Qt GUI application" ON)

# Compiler-specific options
if(MSVC)
    add_compile_options(/utf-8)
    add_compile_options(/W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE MATCHES Release)
        add_compile_options(-O2)
    endif()
endif()

# Find Qt - try Qt6 first, then Qt5 (GUI build only)
set(QT_LIBRARIES "")
if(BUILD_GUI)
    find_package(Qt6 COMPONENTS Widgets QUIET)
    if(Qt6_FOUND)
        message(STATUS "Using Qt6")
        set(QT_VERSION_MAJOR 6)
        set(QT_LIBRARIES Qt6::Widgets)
    else()
        find_package(Qt5 COMPONENTS Widgets QUIET)
        if(Qt5_FOUND)
            message(STATUS "Using Qt5")
            set(QT_VERSION_MAJOR 5)
            set(QT_LIBRARIES Qt5::Widgets)
        endif()
    endif()

    # MSVC cannot build against MinGW/MSYS2 Qt binaries.
    # Detect this early to avoid noisy, misleading compile errors.
    if(MSVC AND (Qt6_FOUND OR Qt5_FOUND))
        set(_qt_target "")
        if(Qt6_FOUND)
            set(_qt_target Qt6::Widgets)
        else()
            set(_qt_target Qt5::Widgets)
        endif()
        get_target_property(_qt_inc_dirs ${_qt_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(_qt_inc_dirs)
            string(JOIN ";" _qt_inc_joined ${_qt_inc_dirs})
            string(TOLOWER "${_qt_inc_joined}" _qt_inc_lower)
            if(_qt_inc_lower MATCHES "msys2" OR _qt_inc_lower MATCHES "mingw")
                message(FATAL_ERROR "Detected MinGW/MSYS2 Qt headers with MSVC compiler. Use the MSYS2/MinGW toolchain (see build.bat), or install a MSVC-built Qt and set Qt5_DIR/Qt6_DIR (or CMAKE_PREFIX_PATH) to that Qt.")
            endif()
        endif()
    endif()
    if(Qt6_FOUND OR Qt5_FOUND)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTORCC ON)
        set(CMAKE_AUTOUIC ON)
    else()
        message(WARNING "Qt not found. GUI target will be skipped. Set Qt6_DIR/Qt5_DIR or CMAKE_PREFIX_PATH to enable.")
        set(BUILD_GUI OFF CACHE BOOL "Build Qt GUI application" FORCE)
    endif()
endif()

# Find threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Interface library for thread dependencies
add_library(project_thread_deps INTERFACE)
target_link_libraries(project_thread_deps INTERFACE Threads::Threads)
if(MINGW)
    target_link_libraries(project_thread_deps INTERFACE winpthread)
endif()

# Collect all source files
file(GLOB_RECURSE PROJECT_SOURCES
    "src/*.cpp"
)

# Collect all header files
file(GLOB_RECURSE PROJECT_HEADERS
    "include/*.h"
)

# Create main executable (GUI)
if(BUILD_GUI)
    add_executable(${PROJECT_NAME}
        ${PROJECT_SOURCES}
        ${PROJECT_HEADERS}
    )

    # Platform-specific executable settings (WIN32 for GUI on Windows)
    if(WIN32)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            WIN32_EXECUTABLE ON
        )
    endif()

    # Include directories
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    # Link libraries
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${QT_LIBRARIES}
        project_thread_deps
    )

    # Install targets
    install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )
else()
    message(STATUS "BUILD_GUI=OFF: skipping NeuralNetworkVisualizer GUI target.")
endif()

# Add tests subdirectory
enable_testing()
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
