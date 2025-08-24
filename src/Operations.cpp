#include "Operations.hpp"
#include "Gemm.hpp"

namespace ember {
namespace Operations {

template <typename scalar_t>
inline void QkvFusedProjection(
    const scalar_t* X, const scalar_t* W_qkv_all,
    scalar_t* QKV,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
)
{
    int rows = batch_size * seq_len;
    int shared_dim = d_model;
    int cols = 3 * d_model;

    ember::Gemm::matmul<scalar_t>(
        X, W_qkv_all, QKV,
        rows, cols, shared_dim
    );
}

template <typename scalar_t>
inline void QkvFusedProjectionBias(
    const scalar_t* X, const scalar_t* W_qkv_all, const scalar_t* bias_qkv_all,
    scalar_t* QKV,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
)
{
    int rows = batch_size * seq_len;
    int shared_dim = d_model;
    int cols = 3 * d_model;

    ember::Gemm::matmul_bias<scalar_t>(
        X, W_qkv_all, bias_qkv_all, QKV,
        rows, cols, shared_dim
    );
}


template <typename scalar_t>
inline void AttentionScores(
    const scalar_t* Q, const scalar_t* K,
    scalar_t* scores,
    int batch_size, int num_heads, int seq_len, int d_head
)
{
    int rows = seq_len;
    int shared_dim = d_head;
    int cols = seq_len;

    ember::Gemm::matmul<scalar_t>(
        Q, K,
        scores,
        batch_size,
        rows, cols, shared_dim,
        false, true,
        static_cast<scalar_t>(1.0 / std::sqrt((float)d_head)),
        static_cast<scalar_t>(0.0)
    );
}

template <typename scalar_t>
inline void AttentionWeightedValues(
    const scalar_t* softmaxScores, const scalar_t* V,
    scalar_t* context,
    int batch_size, int num_heads, int seq_len, int d_head
)
{
    int rows = seq_len;
    int shared_dim = seq_len;
    int cols = d_head;

    ember::Gemm::matmul_strided_batch<scalar_t>(
        softmaxScores, V,
        context,
        batch_size * num_heads,
        rows, cols, shared_dim
    );
}

template <typename scalar_t>
inline void OutputProjection(
    const scalar_t* attentionOutput, const scalar_t* Wo,
    scalar_t* blockOutput,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
)
{
    int rows = batch_size * seq_len;
    int shared_dim = num_heads * d_head;
    int cols = d_model;

    ember::Gemm::matmul<scalar_t>(
        attentionOutput, Wo, blockOutput,
        batch_size * num_heads,
        rows, cols, shared_dim
    );
}

template <typename scalar_t>
inline void OutputProjectionBias(
    const scalar_t* attentionOutput, const scalar_t* Wo, const scalar_t* bias,
    scalar_t* blockOutput,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
)
{
    int rows = batch_size * seq_len;
    int shared_dim = num_heads * d_head;
    int cols = d_model;

    ember::Gemm::matmul_bias<scalar_t>(
        attentionOutput, Wo, bias, blockOutput,
        rows, cols, shared_dim
    );
}





}
}