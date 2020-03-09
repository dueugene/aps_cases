cmake_minimum_required (VERSION 2.8.8)

set(CMAKE_CXX_COMPILER mpic++)

include(./../CMakeInclude.txt)

add_executable(rae rae.cpp)
target_link_libraries(rae ${APSLIB} ${BLASLIBS})

