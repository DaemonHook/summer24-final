cmake_minimum_required(VERSION 3.5)

project(summer24-final)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

enable_language(CUDA)

include_directories("include")

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

aux_source_directory(src SOURCE_FILES)
message("SOURCE_FILES: ${SOURCE_FILES}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_executable(main ${SOURCE_FILES} main.cpp)
target_link_libraries(main ${CUDA_LIBRARIES})

add_executable(test ${SOURCE_FILES} test.cpp)
target_link_libraries(test ${CUDA_LIBRARIES})
