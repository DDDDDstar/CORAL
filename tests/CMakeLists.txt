set(CMAKE_CXX_STANDARD 20)

target_link_libraries(${PROJECT_NAME} PRIVATE
	gtest
	gtest_main
	# mockcpp
	pthread
	)

	# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable


add_executable(test_build_pipeline test_build_pipeline.cpp)
target_link_libraries(test_build_pipeline ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options OpenMP::OpenMP_CXX)


add_executable(test_search_pipeline test_search_pipeline.cpp)
target_link_libraries(test_search_pipeline ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options OpenMP::OpenMP_CXX)

add_executable(comp_gt comp_gt.cpp)
target_link_libraries(comp_gt 
${CUDA_CUDART_LIBRARY}
    faiss
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    OpenMP::OpenMP_CXX)

add_executable(update_compute_gt update_compute_gt.cpp)
target_link_libraries(update_compute_gt 
${CUDA_CUDART_LIBRARY}
    faiss
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    OpenMP::OpenMP_CXX)


add_executable(test_update test_update.cpp)
target_link_libraries(test_update 
${CUDA_CUDART_LIBRARY}
    faiss
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    OpenMP::OpenMP_CXX)
    
add_executable(test_cagra test_cagra.cu)
target_link_libraries(test_cagra 
${CUDA_CUDART_LIBRARY}
    faiss
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    OpenMP::OpenMP_CXX
    rmm::rmm
)
        
add_executable(test_faiss test_faiss.cu)
target_link_libraries(test_faiss 
${CUDA_CUDART_LIBRARY}
    faiss
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    OpenMP::OpenMP_CXX
)
        
add_executable(faiss_gt faiss_gt.cu)
target_link_libraries(faiss_gt 
${CUDA_CUDART_LIBRARY}
    faiss
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    OpenMP::OpenMP_CXX
)
        
add_executable(faiss_pq faiss_pq.cu)
target_link_libraries(faiss_pq 
${CUDA_CUDART_LIBRARY}
    faiss
    ${BLAS_LIBRARIES}
    ${NVML_LIB}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    OpenMP::OpenMP_CXX
)