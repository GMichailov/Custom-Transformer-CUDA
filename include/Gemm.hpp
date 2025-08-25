#include <cublasLt.h>
#include <torch/torch.h>

namespace ember {
namespace Gemm {

extern cublasLtHandle_t handle;
extern cublasLtMatmulDesc_t matmul_desc;
extern cublasLtMatrixLayout_t x_layout, W_qkv_layout, qkv_layout;
cublasLtEpilogue_t relu = CUBLASLT_EPILOGUE_RELU;
cublasLtEpilogue_t relu_bias = CUBLASLT_EPILOGUE_RELU_BIAS;
cublasLtEpilogue_t relu = CUBLASLT_EPILOGUE_GELU;
cublasLtEpilogue_t gelu_bias = CUBLASLT_EPILOGUE_GELU_BIAS;



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
    int strides,
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
