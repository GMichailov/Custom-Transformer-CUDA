#include <cublasLt.h>
#include <torch/torch.h>

namespace ember {
namespace Gemm {

extern cublasLtHandle_t handle;
extern cublasLtMatmulDesc_t desc_plain, desc_bias, desc_bias_relu, desc_bias_gelu;


/*
 * Simple wrappers around cublaslt calls to keep code clean.
 * Function names are clear, so no further documentation required.
 * Built to run on default stream for now. May change this soon.
 */

template <typename scalar_t>
inline void matmul(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int rows, int cols, int shared_dim,
    bool transposeA = false, bool transposeB = false,
    scalar_t scaling_factor = 1.0, scalar_t beta = 0.0
);

template <typename scalar_t>
inline void matmul_bias(
    const scalar_t* A, const scalar_t* B, const scalar_t* bias, scalar_t* C,
    int rows, int cols, int shared_dim,
    bool transposeA = false, bool transposeB = false,
    scalar_t scaling_factor = 1.0, scalar_t beta = 0.0
);

template <typename scalar_t>
inline void matmul_strided_batch(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    long long int strideA, long long int strideB, long long int strideC,
    int rows, int cols, int shared_dim,
    bool transposeA = false, bool transposeB = false,
    scalar_t scaling_factor = 1.0, scalar_t beta = 0.0
);




template <typename scalar_t>
inline void matmul_bias_activation(
    const scalar_t* A, const scalar_t* B, const scalar_t* bias, scalar_t* C,
    int rows, int cols, int shared_dim,
    cublasLtEpilogue_t epilogue
);



template <typename scalar_t>
inline void matmul_activation(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int rows, int cols, int shared_dim,
    cublasLtEpilogue_t epilogue
);




}
}
