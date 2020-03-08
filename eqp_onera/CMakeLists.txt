cmake_minimum_required (VERSION 2.8.8)

set(CMAKE_CXX_COMPILER mpiicc)

include(./../CMakeInclude.txt)

find_package(Boost REQUIRED serialization)

add_executable(onera onera.cpp)
target_link_libraries(onera ${APS_LIB} ${BLAS_LIBS} ${Boost_LIBRARIES})




