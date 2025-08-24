#include "gemm.hpp"

template <typename scalar_t>
void gemm::QkvFusedProjection(
    const scalar_t* X,
    const scalar_t* W_qkv,
    scalar_t* QKV,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
)
{
    int64_t rows = batch_size * seq_len;
    int64_t shared_dim = d_model;
    int64_t cols = 3 * num_heads * d_head;

    float scale = 1.0f / std::sqrt((float)d_model);

    cublasLtMatmul(
        handle,
        matmul_desc,
        &scale,
        X, 
    )
}