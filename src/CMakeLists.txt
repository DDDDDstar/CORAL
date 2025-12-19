file(GLOB_RECURSE CPP_SOURCES *.cpp)
file(GLOB_RECURSE CUDA_SOURCES *.cu)

# 创建共享库
add_library(efanna2e SHARED ${CPP_SOURCES} ${CUDA_SOURCES})

# 设置 target properties
set_target_properties(efanna2e PROPERTIES
    CUDA_ARCHITECTURES 80
    POSITION_INDEPENDENT_CODE ON
)

# 链接库
target_link_libraries(efanna2e PRIVATE
    ${CUDA_CUDART_LIBRARY}
    faiss
    faiss_gpu
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    # ${LZ4_LIBRARY}
    OpenMP::OpenMP_CXX
)