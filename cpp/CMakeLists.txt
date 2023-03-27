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

add_subdirectory(KDTree)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC 
kdtree
benchmark::benchmark 
Catch2::Catch2)