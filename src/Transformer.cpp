#include "Transformer.hpp"

cublasHandle_t handle;

void single_attention_matrix_mul_gemm(
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
    auto Q = torch::empty({batch_size, seq_len, d_model}, X.options());
    auto K = torch::empty({batch_size, seq_len, d_model}, X.options());
    auto V = torch::empty({batch_size, seq_len, d_model}, X.options());

    single_attention_matrix_mul_gemm(X_flattened, Wq, Q);
    single_attention_matrix_mul_gemm(X_flattened, Wk, K);
    single_attention_matrix_mul_gemm(X_flattened, Wv, V);

    // softmax(QK^T / sqrt(D)) * V
    auto scores = torch::empty({batch_size * seq_len, batch_size * seq_len}, X.options());
    single_attention_matrix_mul_gemm(Q, K.transpose(0, 1).contiguous(), scores);
    scores /= std::sqrt((float)d_model);
    auto weights = torch::softmax(scores, -1);
    auto output = torch::empty({batch_size * seq_len, d_model}, X.options());
    single_attention_matrix_mul_gemm(weights, V, output);

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
    // dWq = X^T * dV
    // dWq = X^T * dK

    auto dV = torch::empty_like(V);
    single_attention_matrix_mul_gemm(weights.transpose(1,2), dOutput, dV);
    auto dWeights = torch::empty_like(weights);
    single_attention_matrix_mul_gemm(dOutput, V.transpose(1,2), dWeights);
    auto dScores = torch::_softmax_backward_data(dWeights, weights, -1, dWeights.scalar_type());
    auto dQ = torch::empty_like(Q);
    single_attention_matrix_mul_gemm(dScores, K, dQ);
    auto dK = torch::empty_like(K);
    single_attention_matrix_mul_gemm(dScores.transpose(1,2), Q, dK);

    auto dX_q = torch::empty_like(X);
    auto dX_k = torch::empty_like(X);
    auto dX_v = torch::empty_like(X);

    single_attention_matrix_mul_gemm(dQ, Wq.transpose(0,1), dX_q);
    single_attention_matrix_mul_gemm(dK, Wk.transpose(0,1), dX_k);
    single_attention_matrix_mul_gemm(dV, Wv.transpose(0,1), dX_v);

    dX = dX_q + dX_k + dX_v;

    single_attention_matrix_mul_gemm(X.transpose(0,1), dQ, dWq);
    single_attention_matrix_mul_gemm(X.transpose(0,1), dK, dWk);
    single_attention_matrix_mul_gemm(X.transpose(0,1), dV, dWv);

    return {dX, dWq, dWk, dWv};
}
