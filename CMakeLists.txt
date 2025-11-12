cmake_minimum_required(VERSION 3.10)
project(Atividade_1 C)

set(CMAKE_C_STANDARD 11)

# habilitar o OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# inclui o diretório atual para os .h
include_directories(.)

add_executable(Atividade_1
        main.c
        gaussian_blur.h
        gaussian_blur_parallel.h
)

# link blibioteca da openmp
if(OpenMP_C_FOUND)
    target_link_libraries(Atividade_1 PUBLIC OpenMP::OpenMP_C)
endif()
