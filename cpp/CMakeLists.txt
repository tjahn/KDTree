cmake_minimum_required(VERSION 3.20)
project(KDTree LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD 17)

include(cmake/CPM.cmake)

CPMAddPackage("gh:catchorg/Catch2@3.3.2")
CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.7.1
  OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)

find_package(OpenMP)

if(APPLE)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
        set(OpenMP_C_FLAGS "-fopenmp=libomp")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
        set(OpenMP_C_FLAGS "-fopenmp=libomp")
    endif()

endif()



add_subdirectory(KDTree)

add_subdirectory(benchmarks)

add_subdirectory(PyKDTree)


add_executable(main_benchmark_queries main_benchmark_queries.cpp)
target_link_libraries(main_benchmark_queries PUBLIC 
kdtree
benchmark::benchmark
OpenMP::OpenMP_CXX
)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC 
kdtree
benchmark::benchmark
)
