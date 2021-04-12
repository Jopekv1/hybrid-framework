cmake_minimum_required(VERSION 3.10)
project(hybrid_executor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include_directories(src)
add_subdirectory(third_party)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/load_balancer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/load_balancer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/load_balancer.h
)

set(TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/collatz.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/math.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/max_element.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_pow.cu
)


add_executable(hybrid_executor_tests ${SOURCES} ${TESTS})

set_target_properties(hybrid_executor_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hybrid_executor_tests PRIVATE gtest)