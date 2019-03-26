cmake_minimum_required(VERSION 3.0)

project(csnbvbs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(pybind11)
pybind11_add_module(csnbvbs MODULE src/GRNG.cpp src/InvertY.cpp src/PolyaGamma.cpp src/PolyaGammaAlt.cpp src/PolyaGammaSP.cpp src/RNG.cpp src/pg_sampler.cpp)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})



if(APPLE)
    set(CMAKE_C_COMPILER clang-omp CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER clang-omp++ CACHE STRING "C++ compiler" FORCE)
endif()

target_link_libraries(csnbvbs PUBLIC ${GSL_LIBRARIES})
target_link_libraries(csnbvbs PUBLIC ${EIGEN3_LIBRARIES})

