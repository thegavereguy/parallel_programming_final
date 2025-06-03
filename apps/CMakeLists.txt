add_executable(openmp openmp.cpp)
target_link_libraries(openmp PUBLIC fmt shared)


# ricordarsi di togliare il true forzato
if(DEFINED ENV{BUILD_MPI} AND "$ENV{BUILD_MPI}" STREQUAL "true" OR true)
	message(STATUS "building MPI solution")
	find_package(MPI REQUIRED)
	add_executable(mpi mpi.cpp)
	target_link_libraries(mpi PUBLIC  fmt MPI::MPI_CXX common)
	target_include_directories(mpi PUBLIC ${MPI_CXX_INCLUDE_DIRS})
	message(STATUS "MPI include directories: ${MPI_CXX_INCLUDE_DIRS}")


	# custom target to run mpi target with mpirun ( kind of useless and doesn't work)
	add_custom_target(run_mpi
		COMMAND mpirun -np 4 $<TARGET_FILE:mpi>
		DEPENDS mpi
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Running mpi target with mpirun"
	)
else()
	message("Skipping MPI solution")
endif()
