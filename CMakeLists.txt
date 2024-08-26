cmake_minimum_required(VERSION 3.5)

project(summer24-final)

enable_language(CUDA)

include_directories("include")

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(summer24_SOURCES
    src/main.cpp
	)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_executable(main ${summer24_SOURCES})