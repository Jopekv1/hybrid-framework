cmake_minimum_required(VERSION 3.16)
project(hybrid_executor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/blur_tests.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_add_tests.cu
)


add_executable(hybrid_executor_tests ${SOURCES} ${TESTS})

set_target_properties(hybrid_executor_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP REQUIRED)
target_link_libraries(hybrid_executor_tests PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(hybrid_executor_tests PRIVATE gtest)