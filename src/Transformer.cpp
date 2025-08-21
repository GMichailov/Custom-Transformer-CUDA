#include "Transformer.hpp"

cublasHandle_t handle;

void single_attention_matrix_mul_gemm_2d(
    const torch::Tensor& src_a,
    const torch::Tensor& src_b,
    torch::Tensor& dest
)
{
    auto _src_a = src_a.contiguous();
    auto _src_b = src_b.contiguous();
    auto _dest = dest.contiguous();

    int outer_left_dim = _src_a.size(0);
    int inner_dim = _src_a.size(1);
    int outer_right_dim = _src_b.size(1);

    // cublas gemm does C = alpha * (A*B) + beta C so need alpha=1, beta=0.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cublas does col major calculations, pytorch tensors are row major.
    // AB row major = B^T * A^T col major so do that calculation to avoid transposing.
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // Tells no transpose.
        outer_right_dim, outer_left_dim, inner_dim,
        &alpha,
        _src_b.data_ptr<float>(), outer_right_dim,
        _src_a.data_ptr<float>(), inner_dim,
        &beta,
        _dest.data_ptr<float>(), outer_right_dim
    );

    dest.copy_(_dest);
}

torch::Tensor SingleHeadAttention::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor Wq,
    torch::Tensor Wk,
    torch::Tensor Wv
) 
{
    int batch_size = X.size(0);
    int seq_len = X.size(1);
    int d_model = Wq.size(1);

    // X = <batch_size * seq_len, d_model>
    auto X_flattened = X.view({batch_size * seq_len, X.size(2)});

    // destination tensors
    auto Q = torch::empty({batch_size * seq_len, d_model}, X.options());
    auto K = torch::empty({batch_size * seq_len, d_model}, X.options());
    auto V = torch::empty({batch_size * seq_len, d_model}, X.options());

    single_attention_matrix_mul_gemm_2d(X_flattened, Wq, Q);
    single_attention_matrix_mul_gemm_2d(X_flattened, Wk, K);
    single_attention_matrix_mul_gemm_2d(X_flattened, Wv, V);

    // softmax(QK^T / sqrt(D)) * V
    auto scores = torch::empty({batch_size * seq_len, batch_size * seq_len}, X.options());
    single_attention_matrix_mul_gemm_2d(Q, K.transpose(0, 1).contiguous(), scores);
    scores /= std::sqrt((float)d_model);
    auto weights = torch::softmax(scores, -1);
    auto output = torch::empty({batch_size * seq_len, d_model}, X.options());
    single_attention_matrix_mul_gemm_2d(weights, V, output);

    // For now, save everything, can do recomputes later when I have time.
    ctx->save_for_backward({X, Wq, Wk, Wv, Q, K, V, weights});
    
    // Reshape to original dimesnions
    return output.view({batch_size, seq_len, d_model});
}

torch::autograd::tensor_list SingleHeadAttention::backward(
    torch::autograd::AutogradContext *ctx, 
    torch::autograd::tensor_list grad_outputs
)
{
    // Backprop
    auto dOutput = grad_outputs[0];
    auto forward_pass_vars = ctx->get_saved_variables();
    auto X  = forward_pass_vars[0];
    auto Wq = forward_pass_vars[1];
    auto Wk = forward_pass_vars[2];
    auto Wv = forward_pass_vars[3];
    auto Q  = forward_pass_vars[4];
    auto K  = forward_pass_vars[5];
    auto V  = forward_pass_vars[6];
    auto weights = forward_pass_vars[7];

    int batch_size = X.size(0);
    int seq_len    = X.size(1);
    int d_model    = Wq.size(1);

    // Create flattened pointers because my function can only do 2D math right now.
    auto X_flat       = X.view({batch_size * seq_len, X.size(2)});
    auto dOutput_flat = dOutput.view({batch_size * seq_len, d_model});
    auto Q_flat       = Q.view({batch_size * seq_len, d_model});
    auto K_flat       = K.view({batch_size * seq_len, d_model});
    auto V_flat       = V.view({batch_size * seq_len, d_model});
    auto weights_flat = weights.view({batch_size * seq_len, batch_size * seq_len});

    // Create gradient tensor containers
    auto dX  = torch::empty_like(X);
    auto dWq = torch::empty_like(Wq);
    auto dWk = torch::empty_like(Wk);
    auto dWv = torch::empty_like(Wv);

    // dV = W^T * dOutput
    // dWeights = dOutput * V^T
    // dScores = torch::softmax_backward(dWeights, weights).
    // dQ = dScores * K
    // dK = dScores ^ T * Q
    // dX = dQ * Wq^T + dk * Wk^T + dV * Wv^T
    // dWq = X^T * dQ
    // dWk = X^T * dK
    // dWv = X^T * dV

    auto dV = torch::empty_like(V_flat);
    single_attention_matrix_mul_gemm_2d(weights.transpose(0,1).contiguous(), dOutput_flat, dV);
    auto dWeights = torch::empty_like(weights_flat);
    single_attention_matrix_mul_gemm_2d(dOutput, V_flat.transpose(0,1).contiguous(), dWeights);
    auto dScores = torch::_softmax_backward_data(dWeights, weights_flat, -1, dWeights.scalar_type());
    auto dQ = torch::empty_like(Q_flat);
    single_attention_matrix_mul_gemm_2d(dScores, K_flat, dQ);
    auto dK = torch::empty_like(K_flat);
    single_attention_matrix_mul_gemm_2d(dScores.transpose(0,1), Q_flat, dK);

    auto dX_q = torch::empty_like(X_flat);
    auto dX_k = torch::empty_like(X_flat);
    auto dX_v = torch::empty_like(X_flat);

    single_attention_matrix_mul_gemm_2d(dQ, Wq.transpose(0,1), dX_q);
    single_attention_matrix_mul_gemm_2d(dK, Wk.transpose(0,1), dX_k);
    single_attention_matrix_mul_gemm_2d(dV, Wv.transpose(0,1), dX_v);

    dX.copy_(dX_q).add_(dX_k).add_(dX_v);

    single_attention_matrix_mul_gemm_2d(X_flat.transpose(0,1), dQ, dWq);
    single_attention_matrix_mul_gemm_2d(X_flat.transpose(0,1), dK, dWk);
    single_attention_matrix_mul_gemm_2d(X_flat.transpose(0,1), dV, dWv);

    return {dX.view_as(X), dWq, dWk, dWv};
}
