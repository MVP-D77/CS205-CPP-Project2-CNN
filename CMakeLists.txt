cmake_minimum_required(VERSION 3.14)
project(final_project)
#openCV库
find_package(OpenCV REQUIRED)

#将mac自带的clang替换为gcc-10编译器，从而使用多线程
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_C_COMPILER "/usr/local/Cellar/gcc/10.2.0/bin/gcc-10")
#set(CMAKE_CXX_COMPILER "/usr/local/Cellar/gcc/10.2.0/bin/g++-10")

#omp
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
endif()

#openBlas 库
include_directories("/usr/local/Cellar/openblas/0.3.10_2/include")
link_directories("/usr/local/Cellar/openblas/0.3.10_2/lib")
link_libraries(cblas)

#指令集
add_compile_options(-march=native)
#C++11标准以及指令集
add_compile_options(-std=c++11 -O3)

add_executable(final_project main.cpp Matrix.cpp)
target_link_libraries(final_project ${OpenCV_LIBS})

