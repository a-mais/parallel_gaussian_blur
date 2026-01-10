cmake_minimum_required(VERSION 3.10)
project(Atividade_1 C)

set(CMAKE_C_STANDARD 11)

# MPI first (with hint for MS-MPI SDK)
set(MPI_DIR "C:/Program Files (x86)/Microsoft SDKs/MPI")
find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_C_LIBRARIES}")

# OpenMP
find_package(OpenMP REQUIRED)

include_directories(.)

add_executable(Atividade_1
        main.c
        gaussian_blur.h
        gaussian_blur_parallel.h
)

target_link_libraries(Atividade_1 PUBLIC OpenMP::OpenMP_C MPI::MPI_C)

target_compile_definitions(Atividade_1 PRIVATE USE_MPI)
