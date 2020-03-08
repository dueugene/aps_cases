cmake_minimum_required (VERSION 2.8.8)

set(CMAKE_CXX_COMPILER mpiicc)

include(./../CMakeInclude.txt)

find_package(Boost REQUIRED serialization)

add_executable(naca_euler_eqp naca_euler_eqp.cpp)
target_link_libraries(naca_euler_eqp ${APSLIB} ${BLAS_LIBS} ${Boost_LIBRARIES})



