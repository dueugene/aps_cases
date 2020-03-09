cmake_minimum_required (VERSION 2.8.8)

set(CMAKE_CXX_COMPILER mpic++)

include(./../CMakeInclude.txt)

find_package(Boost REQUIRED serialization)

add_executable(naca_eqp naca_eqp.cpp)
target_link_libraries(naca_eqp ${APSLIB} ${BLAS_LIBS} ${Boost_LIBRARIES})





