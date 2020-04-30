cmake_minimum_required (VERSION 2.8.8)

set(CMAKE_CXX_COMPILER mpic++)

include(./../CMakeInclude.txt)

find_package(Boost REQUIRED serialization)

add_executable(blender_test blender_test.cpp)
target_link_libraries(blender_test ${APSLIB} ${BLAS_LIBS} ${Boost_LIBRARIES})



