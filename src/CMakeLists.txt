
file(GLOB_RECURSE CPP_SOURCES *.cpp)
file(GLOB_RECURSE CUDA_SOURCES *.cu)

# 查找BLAS库
find_package(BLAS REQUIRED)
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")

find_library(NVML_LIB 
    NAMES nvidia-ml 
    PATHS /usr/lib/x86_64-linux-gnu
    REQUIRED)

add_library(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_ARCHITECTURES "80"
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${PROJECT_NAME} 
    ${BLAS_LIBRARIES} 
    ${CUDA_LIBRARIES} 
    ${NVML_LIB})

add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES} ${CUDA_SOURCES})

set_target_properties(${PROJECT_NAME}_s PROPERTIES
  CUDA_ARCHITECTURES "80"
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${PROJECT_NAME}_s 
    ${BLAS_LIBRARIES} 
    ${CUDA_LIBRARIES} 
    ${NVML_LIB})
