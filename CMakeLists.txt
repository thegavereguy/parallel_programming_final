cmake_minimum_required(VERSION 3.10)
project(ParalleProgrammingFilan CXX)

set(CMAKE_CXX_STANDARD 17)

Include(FetchContent)

add_subdirectory(third_party)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tests)
