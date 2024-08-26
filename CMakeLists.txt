cmake_minimum_required(VERSION 3.5)

project(summer24-final)

enable_language(CUDA)

include_directories("include")

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(SOURCE_FILES
    src/Graph.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_executable(main ${SOURCE_FILES} main.cpp)

add_executable(test ${SOURCE_FILES} test.cpp)
