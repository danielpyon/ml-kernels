cmake_minimum_required(VERSION 3.23.1)
project(ml-kernels LANGUAGES CXX CUDA VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TARGET_NAME tutorial)
set(PROJECT_BINARY_DIR build)

add_executable(${TARGET_NAME} tutorial.cc)
add_subdirectory(MathFunctions)
add_subdirectory(kernels)
target_link_libraries(${TARGET_NAME} PUBLIC MathFunctions)
target_link_libraries(${TARGET_NAME} PUBLIC kernels)

configure_file(TutorialConfig.h.in TutorialConfig.h)
# header files
target_include_directories(${TARGET_NAME} PUBLIC
    ${PROJECT_BINARY_DIR}
    MathFunctions
    kernels)
