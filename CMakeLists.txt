cmake_minimum_required(VERSION 3.17)

project(efanna2e
    VERSION 1.0
    LANGUAGES CXX CUDA
)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda-11.4/include)
# link_directories(${CUDA_LIBRARY_DIRS}) 

enable_language(CUDA)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/cub)

find_package(Boost COMPONENTS program_options REQUIRED)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIR})
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -Wall -Wextra -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -mtune=native -ftree-vectorize -Wall -DINFO")

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "80")

add_subdirectory(src)
add_subdirectory(tests)
