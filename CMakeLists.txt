cmake_minimum_required(VERSION 3.23.1)
project(ml-kernels VERSION 1.0)

# Check for the presence of NVIDIA GPU
find_program(NVCC nvcc HINTS /usr/local/cuda)
if (NVCC)
    message(STATUS "NVIDIA GPU detected, using CUDA")
    enable_language(CUDA)
    set(GPU_LANGUAGE "CUDA")
else()
    # Check for the presence of an AMD GPU (HIP)
    find_program(HIPCC hipcc HINTS /opt/rocm/bin)
    if (HIPCC)
        message(STATUS "AMD GPU detected, using HIP")
        # set(CMAKE_HIP_COMPILER ${HIPCC})
        enable_language(HIP)
        set(GPU_LANGUAGE "HIP")
    else()
        message(FATAL_ERROR "No compatible GPU found (neither NVIDIA nor AMD)")
    endif()
endif()

# Set project languages based on the detected GPU
if (GPU_LANGUAGE STREQUAL "CUDA")
    project(ml-kernels LANGUAGES CXX CUDA)
elseif (GPU_LANGUAGE STREQUAL "HIP")
    project(ml-kernels LANGUAGES CXX HIP)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TARGET_NAME test)
set(PROJECT_BINARY_DIR build)

add_subdirectory(3rdparty/googletest)

add_executable(${TARGET_NAME} test.cpp)

# link libs
add_subdirectory(kernels)
target_link_libraries(${TARGET_NAME}
    PUBLIC
    kernels
    gtest
    gtest_main
)

# header files
target_include_directories(${TARGET_NAME} PUBLIC
    ${PROJECT_BINARY_DIR}
    kernels)
