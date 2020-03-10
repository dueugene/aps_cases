cmake_minimum_required (VERSION 2.8.8)

set(CMAKE_CXX_COMPILER mpic++)

include(./../CMakeInclude.txt)

find_package(Boost REQUIRED serialization)

add_executable(rae rae.cpp)
target_link_libraries(rae ${APSLIB} ${BLAS_LIBS} ${Boost_LIBRARIES})

