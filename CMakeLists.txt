cmake_minimum_required(VERSION 3.14)
project(hpdp)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

file(GLOB SOURCES
    "src/*.h"
    "src/*.cpp"
)

add_executable(hpdp ${SOURCES})

find_package(OpenMP REQUIRED)
target_link_libraries(hpdp PRIVATE OpenMP::OpenMP_CXX)