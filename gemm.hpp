#include <cublasLt.h>
#include <torch/torch.h>

namespace gemm
{

extern cublasLtHandle_t handle;
extern cublasLtMatmulDesc_t matmul_desc;
extern cublasLtMatrixLayout_t x_layout, W_qkv_layout, qkv_layout;
extern cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;

// Template for handling mixed precision.
template <typename scalar_t>
inline void QkvFusedProjection(
    const scalar_t* X,
    const scalar_t* W_qkv,
    scalar_t* QKV,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
);

inline void FlashAttention();

inline void OutputProjection();

}
