cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(single_head_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/opt/libtorch")

find_package(Torch REQUIRED)

# --- Find CUDA libraries explicitly ---
find_library(CUBLAS_LIB cublas HINTS /usr/local/cuda-12.9/lib64)
find_library(CUBLASLT_LIB cublasLt HINTS /usr/local/cuda-12.9/lib64)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda-12.9/lib64)

if(NOT CUBLAS_LIB)
    message(FATAL_ERROR "Could not find libcublas.so in /usr/local/cuda-12.9/lib64")
endif()
if(NOT CUBLASLT_LIB)
    message(FATAL_ERROR "Could not find libcublasLt.so in /usr/local/cuda-12.9/lib64")
endif()
if(NOT CUDART_LIB)
    message(FATAL_ERROR "Could not find libcudart.so in /usr/local/cuda-12.9/lib64")
endif()

# --- Executable ---
add_executable(single_head_attention_test
    main.cpp
    Transformer.cpp
)

# --- Includes ---
target_include_directories(single_head_attention_test PRIVATE
    /opt/libtorch/include
    /opt/libtorch/include/torch/csrc/api/include
    /usr/local/cuda-12.9/include
)

# --- Linking ---
target_link_libraries(single_head_attention_test
    "${TORCH_LIBRARIES}"
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDART_LIB}
)

set_property(TARGET single_head_attention_test PROPERTY CXX_STANDARD 17)
