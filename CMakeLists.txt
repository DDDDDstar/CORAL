cmake_minimum_required(VERSION 3.18)
project(efanna2e VERSION 1.0 LANGUAGES CXX CUDA)

cmake_policy(SET CMP0079 NEW)

# --------------------------
# C++ / CUDA 设置
# --------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# set(CMAKE_IGNORE_PATH "/opt/conda/lib")
# 指定系统 CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
find_package(CUDA REQUIRED)   # 使用旧式 FindCUDA，保证 cudart 能找到
add_library(CUDA::cudart UNKNOWN IMPORTED)
set_target_properties(CUDA::cudart PROPERTIES
  IMPORTED_LOCATION "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_ROOT_DIR}/include"
)
add_library(CUDA::cublas UNKNOWN IMPORTED)
set_target_properties(CUDA::cublas PROPERTIES
  IMPORTED_LOCATION "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_ROOT_DIR}/include"
)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

enable_language(CUDA)

# --------------------------
# 依赖库
# --------------------------
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Faiss REQUIRED)
find_package(BLAS REQUIRED)

# LZ4
# find_path(LZ4_INCLUDE_DIR lz4.h REQUIRED)
# find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

# NVML
find_library(NVML_LIB NAMES nvidia-ml PATHS /usr/lib/x86_64-linux-gnu REQUIRED)

# include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${FAISS_INCLUDE_DIRS}
    # ${LZ4_INCLUDE_DIR}
)

# --------------------------
# 编译选项
# --------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -Wall -Wextra -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -mtune=native -ftree-vectorize -Wall -DINFO")

# --------------------------
# 子目录
# --------------------------
add_subdirectory(src)
add_subdirectory(tests)