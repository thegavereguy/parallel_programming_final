add_executable(cpu_bench cpu_bench.cpp)
add_executable(cpu_tests cpu_tests.cpp)

target_link_libraries(cpu_bench PUBLIC fmt Catch2::Catch2 shared )
target_link_libraries(cpu_tests PUBLIC fmt Catch2::Catch2WithMain shared common )

if(DEFINED ENV{BUILD_MPI} AND "$ENV{BUILD_MPI}" STREQUAL "true")
	# add_executable(mpi_bench mpi_bench.cpp)
	# target_link_libraries(mpi_bench PUBLIC fmt Catch2::Catch2WithMain common)
	# target_include_directories(mpi_bench PUBLIC ${MPI_CXX_INCLUDE_PATH})
endif()


