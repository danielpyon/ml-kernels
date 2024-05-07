cmake_minimum_required(VERSION 3.23.1)

# Add the include directory to the include search path
include_directories(../include)

# Add source files
add_executable(test_sampling test_sampling.cpp)

# Link your executable with the library
target_link_libraries(test_sampling libmlk)
