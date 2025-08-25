#pragma once
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace ember {
namespace Precision {

enum class Precision { FP32, FP16, BF16 };

template <typename T>
struct CublasLtType;

template <> struct CublasLtType<float> { 
    static constexpr cudaDataType_t type = CUDA_R_32F; 
    static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};
template <> struct CublasLtType<__half> { 
    static constexpr cudaDataType_t type = CUDA_R_16F; 
    static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_16F;
};
template <> struct CublasLtType<__nv_bfloat16> { 
    static constexpr cudaDataType_t type = CUDA_R_16BF; 
    static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};
template <> struct CublasLtType<int8_t> { 
    static constexpr cudaDataType_t type = CUDA_R_8I; 
    static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32I;
};


}
}