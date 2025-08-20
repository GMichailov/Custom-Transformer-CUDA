cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(single_head_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/opt/libtorch")

find_package(Torch REQUIRED)

find_library(CUBLAS_LIB cublas HINTS /usr/local/cuda-12.9/lib64)
find_library(CUBLASLT_LIB cublasLt HINTS /usr/local/cuda-12.9/lib64)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda-12.9/lib64)

add_executable(single_head_attention_test
    main.cpp
    Transformer.cpp
)

target_include_directories(single_head_attention_test PRIVATE
    /opt/libtorch/include
    /opt/libtorch/include/torch/csrc/api/include
    /usr/local/cuda-12.9/include
)

target_link_libraries(single_head_attention_test
    "${TORCH_LIBRARIES}"
    cublas
    cublasLt
    cuda
)

set_property(TARGET single_head_attention_test PROPERTY CXX_STANDARD 17)