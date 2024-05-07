cmake_minimum_required(VERSION 3.23.1)

if(BUILD_TESTING)

  add_executable(test_sampling test_sampling.cu)

  set_target_properties(particle_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(particle_test PRIVATE particles)

  add_test(NAME test-triton COMMAND python3 test.py)

  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()
