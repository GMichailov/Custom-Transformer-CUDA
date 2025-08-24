#include <cublasLt.h>
#include <torch/torch.h>

namespace ember {
namespace Gemm {

extern cublasLtHandle_t handle;
extern cublasLtMatmulDesc_t matmul_desc;
extern cublasLtMatrixLayout_t x_layout, W_qkv_layout, qkv_layout;
extern cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;


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
inline void matmul_bias_gelu(
    const scalar_t* A, const scalar_t* B, scalar_t bias, scalar_t* C,
    int rows, int cols, int shared_dim
);

template <typename scalar_t>
inline void matmul_strided_batch(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int strides,
    int rows, int cols, int shared_dim,
    bool transposeA = false, bool transposeB = false,
    scalar_t scaling_factor = 1.0, scalar_t beta = 0.0
);

}
}
