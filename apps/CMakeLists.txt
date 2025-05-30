add_executable(openmp openmp.cpp)
target_link_libraries(openmp PUBLIC  fmt shared)


if(DEFINED ENV{BUILD_MPI} AND "$ENV{BUILD_MPI}" STREQUAL "true" OR true)
	message(STATUS "building MPI solution")
	find_package(MPI REQUIRED)
	add_executable(mpi mpi.cpp)
	target_link_libraries(mpi PUBLIC  fmt MPI::MPI_CXX common)
	target_include_directories(mpi PUBLIC ${MPI_CXX_INCLUDE_PATH})

# if(DEFINED ENV{BUILD_OPTIMIZED} AND "$ENV{BUILD_OPTIMIZED}" STREQUAL "true")
# 		message("Building optimized version of MPI")
# 	target_compile_options(mpi PUBLIC -O3)
# else()
# 	target_compile_options(mpi PUBLIC -O0)
# 		message("Building non-optimized version of MPI")
# endif()

# custom target to run mpi target with mpirun
	add_custom_target(run_mpi
		COMMAND mpirun -np 4 $<TARGET_FILE:mpi>
		DEPENDS mpi
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Running mpi target with mpirun"
	)
else()
	message("Skipping MPI solution")
endif()
