set(CMAKE_CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME}
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
target_link_libraries(test_build_pipeline ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)


add_executable(test_search_pipeline test_search_pipeline.cpp)
target_link_libraries(test_search_pipeline ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)
