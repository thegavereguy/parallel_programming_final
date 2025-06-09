cmake_minimum_required(VERSION 3.10)
project(ParalleProgrammingFilan CXX)

set(CMAKE_CXX_STANDARD 17)

Include(FetchContent)

option(ENABLE_OPTIMIZATION "Enable compiler optimizations" ON)
set(OPTIMIZATION_LEVEL "O3" CACHE STRING "Compiler optimization level (e.g., O0, O1, O2, O3, Os, Ofast)")

if(NOT CMAKE_BUILD_TYPE)
    # default to Release if CMAKE_BUILD_TYPE is not set externally
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(ENABLE_OPTIMIZATION)
			# string(TOUPPER "${OPTIMIZATION_LEVEL}" OPTIMIZATION_LEVEL_UPPER) # rompe tutto quando si usa -Ofast
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -${OPTIMIZATION_LEVEL}")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -${OPTIMIZATION_LEVEL}")
        set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -${OPTIMIZATION_LEVEL}")
        message(STATUS "Using optimization level: -${OPTIMIZATION_LEVEL}")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O0")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O0")
        set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -O0")
        message(STATUS "Optimizations disabled: -O0")
    endif()

    # Aggiungi altre flag comuni per ottimizzazione e debug se necessario
	# TODO: provare a togliere durante testing
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG") 
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
endif()

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tests)
add_subdirectory(third_party)
