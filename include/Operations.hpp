#pragma once
#include "Gemm.hpp"

namespace ember {
namespace Operations {

/*
 * Expects:
 * X : [batch_size, seq_len, d_model]
 * W_qkv_all : [d_model, 3*d_model] (concatenated [Wq | Wk | Wv])
 * QKV: [batch*seq_len, 3*d_model]
 * 
 * Caller must reshape QKV into:
 * Q: [batch_size, num_heads, seq_len, d_head]
 * K: [batch_size, num_heads, seq_len, d_head]
 * V: [batch_size, num_heads, seq_len, d_head]
 */
template <typename scalar_t>
inline void QkvFlatFusedProjection(
    const scalar_t* X, const scalar_t* W_qkv_all,
    scalar_t* QKV,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
);




/*
 * Expects:
 * X : [batch_size, seq_len, d_model]
 * W_qkv_all : [d_model, 3*d_model] (concatenated [Wq | Wk | Wv])
 * b_qkv_all : [3*d_model] (concatenated [bq | bk | bv])
 * QKV: [batch*seq_len, 3*d_model]
 * 
 * Caller must reshape QKV into:
 * Q: [batch_size, num_heads, seq_len, d_head]
 * K: [batch_size, num_heads, seq_len, d_head]
 * V: [batch_size, num_heads, seq_len, d_head]
 */
template <typename scalar_t>
inline void QkvFlatFusedProjectionBias(
    const scalar_t* X, const scalar_t* W_qkv_all, const scalar_t* bias_qkv_all,
    scalar_t* QKV,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
);




/*
 * Computes scaled dot-product attention scores across batches and heads.
 *
 * Expects:
 * Q: [batch_size, num_heads, seq_len, d_head]
 * K: [batch_size, num_heads, seq_len, d_head]
 * scores : [batch_size, num_heads, seq_len, seq_len]
 *
 */
template <typename scalar_t>
inline void AttentionScores(
    const scalar_t* Q, const scalar_t* K,
    scalar_t* scores,
    int batch_size, int num_heads, int seq_len, int d_head
);



/*
 * Computes attention-weighted values.
 *
 * Expects:
 * softmaxScores: [batch_size, num_heads, seq_len, seq_len]
 * V: [batch_size, num_heads, seq_len, d_head]
 * context : [batch_size, num_heads, seq_len, d_head]
 *
 */
template <typename scalar_t>
inline void AttentionWeightedValues(
    const scalar_t* softmaxScores, const scalar_t* V,
    scalar_t* context,
    int batch_size, int num_heads, int seq_len, int d_head
);




/*
 * Final output projection after multi-head attention.
 *
 * Expects:
 * context : [batch_size, num_heads, seq_len, d_head]
 * Wo : [num_heads * d_head, d_model]
 * outputProjection : [batch_size, seq_len, d_model]
 *
 */
template <typename scalar_t>
inline void OutputProjection(
    const scalar_t* context, const scalar_t* Wo,
    scalar_t* outputProjection,
    int batch_size, int seq_len, int d_model, int num_heads, int d_head
);





/*
 * Final output projection after multi-head attention.
 *
 * Expects:
 * context : [batch_size, num_heads, seq_len, d_head]
 * Wo : [num_heads * d_head, d_model]
 * bo : [d_model]
 * outputProjection : [batch_size, seq_len, d_model]
 *
 */
template <typename scalar_t>
inline void OutputProjectionBias(
    const scalar_t* attentionOutput, const scalar_t* Wo, const scalar_t* bo,
    scalar_t* blockOutput,
    int batch_size, int seq_len, int d_model, int head, int d_head
);

template <typename scalar_t>
inline void FeedForwardActivation(
    const scalar_t* X, const scalar_t* W,
    scalar_t* output
);

template <typename scalar_t>
inline void FeedForwardBiasActivation(
    const scalar_t* X, const scalar_t* W, const scalar_t* b,
    scalar_t* output
);

template <typename scalar_t>
inline void FeedForwardNoActivation(
    const scalar_t* hidden, const scalar_t* W,
    scalar_t* output
);

template <typename scalar_t>
inline void FeedForwardBiasNoActivation(
    const scalar_t* hidden, const scalar_t* W, const scalar_t* b,
    scalar_t* output
);

// TODO Later: Import FlashAttention/Rewrite anything that uses python.h

}
}