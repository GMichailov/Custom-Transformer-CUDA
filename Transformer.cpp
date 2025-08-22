#include "Transformer.hpp"

cublasHandle_t handle;

// Calculation Functions

void gemm_2d(
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

void gemm_batched(
    const torch::Tensor& src_a,
    const torch::Tensor& src_b,
    torch::Tensor& dest,
    int batchCount
)
{
    auto _src_a = src_a.contiguous();
    auto _src_b = src_b.contiguous();
    auto _dest = dest.contiguous();

    int outer_left_dim = _src_a.size(1);
    int inner_dim = _src_a.size(2);
    int outer_right_dim = _src_b.size(2);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    long long stride_a = (long long)outer_left_dim * inner_dim;
    long long stride_b = (long long)inner_dim * outer_right_dim;
    long long stride_dest = (long long)outer_left_dim * outer_right_dim;

    // Same swap as in 2D mul
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        outer_right_dim, outer_left_dim, inner_dim,
        &alpha,
        _src_b.data_ptr<float>(), outer_right_dim, stride_b,
        _src_a.data_ptr<float>(), outer_left_dim, stride_a,
        &beta,
        _dest.data_ptr<float>(), outer_right_dim, stride_dest,
        batchCount
    );

    dest.copy_(_dest);
}

// Positional Encoders

// Attention

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

    gemm_2d(X_flattened, Wq, Q);
    gemm_2d(X_flattened, Wk, K);
    gemm_2d(X_flattened, Wv, V);

    // softmax(QK^T / sqrt(D)) * V
    auto scores = torch::empty({batch_size * seq_len, batch_size * seq_len}, X.options());
    gemm_2d(Q, K.transpose(0, 1).contiguous(), scores);
    scores /= std::sqrt((float)d_model);
    auto weights = torch::softmax(scores, -1);
    auto output = torch::empty({batch_size * seq_len, d_model}, X.options());
    gemm_2d(weights, V, output);

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
    gemm_2d(weights.transpose(0,1).contiguous(), dOutput_flat, dV);
    auto dWeights = torch::empty_like(weights_flat);
    gemm_2d(dOutput, V_flat.transpose(0,1).contiguous(), dWeights);
    auto dScores = torch::_softmax_backward_data(dWeights, weights_flat, -1, dWeights.scalar_type());
    auto dQ = torch::empty_like(Q_flat);
    gemm_2d(dScores, K_flat, dQ);
    auto dK = torch::empty_like(K_flat);
    gemm_2d(dScores.transpose(0,1), Q_flat, dK);

    auto dX_q = torch::empty_like(X_flat);
    auto dX_k = torch::empty_like(X_flat);
    auto dX_v = torch::empty_like(X_flat);

    gemm_2d(dQ, Wq.transpose(0,1), dX_q);
    gemm_2d(dK, Wk.transpose(0,1), dX_k);
    gemm_2d(dV, Wv.transpose(0,1), dX_v);

    dX.copy_(dX_q).add_(dX_k).add_(dX_v);

    gemm_2d(X_flat.transpose(0,1), dQ, dWq);
    gemm_2d(X_flat.transpose(0,1), dK, dWk);
    gemm_2d(X_flat.transpose(0,1), dV, dWv);

    return {dX.view_as(X), dWq, dWk, dWv};
}

torch::Tensor MultiHeadAttention::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor Wq,
    torch::Tensor Wk,
    torch::Tensor Wv,
    torch::Tensor Wo,
    int num_heads
)
{
    int batch_size = X.size(0);
    int seq_len = X.size(1);
    int d_model = Wq.size(0);
    int head_dim = d_model / num_heads;

    auto X_flattened = X.view({batch_size * seq_len, d_model});

    auto Q_all = torch::empty({batch_size * seq_len, d_model}, X.options());
    auto K_all = torch::empty({batch_size * seq_len, d_model}, X.options());
    auto V_all = torch::empty({batch_size * seq_len, d_model}, X.options());

    gemm_2d(X_flattened, Wq, Q_all);
    gemm_2d(X_flattened, Wk, K_all);
    gemm_2d(X_flattened, Wv, V_all);

    auto Q = Q_all.view({batch_size, seq_len, num_heads, head_dim}).permute({0,2,1,3}).contiguous();
    auto K = K_all.view({batch_size, seq_len, num_heads, head_dim}).permute({0,2,1,3}).contiguous();
    auto V = V_all.view({batch_size, seq_len, num_heads, head_dim}).permute({0,2,1,3}).contiguous();

    auto scores = torch::empty({batch_size, num_heads, seq_len, seq_len}, X.options());
    gemm_batched(Q, K.transpose(-2, -1).contiguous(), scores, batch_size * num_heads);
    scores /= std::sqrt((float)head_dim);
    auto weights = torch::softmax(scores, -1);

    auto output = torch::empty({batch_size, num_heads, seq_len, head_dim}, X.options());
    gemm_batched(weights, V, output, batch_size * num_heads);

    auto concatenated = output.permute({0,2,1,3}).contiguous().view({batch_size, seq_len, d_model});
    auto final_result = torch::empty_like(concatenated);
    gemm_2d(concatenated, Wo, final_result);

    ctx->save_for_backward({X, Wq, Wk, Wv, Wo, Q, K, V, weights, concatenated});
    return final_result.view({batch_size, seq_len, d_model});
}

torch::autograd::tensor_list MultiHeadAttention::backward(
    torch::autograd::AutogradContext *ctx, 
    torch::autograd::tensor_list grad_outputs
) 
{
    auto dFinal_result = grad_outputs[0];
    auto forward_pass_vars = ctx->get_saved_variables();
    auto X = forward_pass_vars[0];
    auto Wq = forward_pass_vars[1];
    auto Wk = forward_pass_vars[2];
    auto Wv = forward_pass_vars[3];
    auto Wo  = forward_pass_vars[4];
    auto Q   = forward_pass_vars[5];
    auto K   = forward_pass_vars[6];
    auto V   = forward_pass_vars[7];
    auto weights = forward_pass_vars[8];
    auto concatenated  = forward_pass_vars[9];

    int batch_size = X.size(0);
    int seq_len = X.size(1);
    int d_model = Wq.size(0);
    int num_heads = Q.size(1);
    int head_dim = d_model / num_heads;

    auto dFinal_result_flat = dFinal_result.view({batch_size * seq_len, d_model});
    auto concatenated_flat = concatenated.view({batch_size * seq_len, d_model});
    auto dWo = torch::empty_like(Wo);
    auto dConcatenated = torch::empty_like(concatenated);
    gemm_2d(concatenated_flat.transpose(0,1).contiguous(), dFinal_result_flat, dWo);
    gemm_2d(dFinal_result_flat, Wo.transpose(0,1).contiguous(), dConcatenated);
    auto dConcatenated_heads = dConcatenated.view({batch_size, seq_len, num_heads, head_dim}).permute({0,2,1,3}).contiguous();


    auto Q_flat = Q.view({batch_size * num_heads, seq_len, head_dim});
    auto K_flat = K.view({batch_size * num_heads, seq_len, head_dim});
    auto V_flat = V.view({batch_size * num_heads, seq_len, head_dim});
    auto weights_flat = weights.view({batch_size * num_heads, seq_len, seq_len});
    auto dOutput_flat = dConcatenated_heads.view({batch_size * num_heads, seq_len, head_dim});

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
    gemm_batched(weights.transpose(1,2).contiguous(), dOutput_flat, dV, batch_size*num_heads);
    auto dWeights = torch::empty_like(weights_flat);
    gemm_batched(dOutput_flat, V_flat.transpose(1,2).contiguous(), dWeights, batch_size*num_heads);
    auto dScores = torch::_softmax_backward_data(dWeights, weights_flat, -1, dWeights.scalar_type());
    auto dQ = torch::empty_like(Q_flat);
    gemm_batched(dScores, K_flat, dQ, batch_size*num_heads);
    auto dK = torch::empty_like(K_flat);
    gemm_batched(dScores.transpose(1,2).contiguous(), Q_flat, dK, batch_size*num_heads);

    auto X_flat = X.view({batch_size * seq_len, d_model});

    auto dQ_merge = dQ.view({batch_size * seq_len, d_model});
    auto dK_merge = dK.view({batch_size * seq_len, d_model});
    auto dV_merge = dV.view({batch_size * seq_len, d_model});

    auto dX_q = torch::empty_like(X_flat);
    auto dX_k = torch::empty_like(X_flat);
    auto dX_v = torch::empty_like(X_flat);

    gemm_2d(dQ_merge, Wq.transpose(0,1).contiguous(), dX_q);
    gemm_2d(dK_merge, Wk.transpose(0,1).contiguous(), dX_k);
    gemm_2d(dV_merge, Wv.transpose(0,1).contiguous(), dX_v);

    dX.copy_(dX_q).add_(dX_k).add_(dX_v);

    gemm_2d(X_flat.transpose(0,1).contiguous(), dQ_merge, dWq);
    gemm_2d(X_flat.transpose(0,1).contiguous(), dK_merge, dWk);
    gemm_2d(X_flat.transpose(0,1).contiguous(), dV_merge, dWv);

    return {dX.view_as(X), dWq, dWk, dWv, dWo, torch::Tensor()};
}

// MLP

torch::Tensor Linear::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor b
)
{
    auto prebias_output = torch::empty({X.size(0), W.size(1)}, X.options());
    gemm_2d(X, W, prebias_output);
    auto output = prebias_output + b;
    ctx->save_for_backward({X, W, b});
    return output;
}

torch::autograd::tensor_list Linear::backward(
    torch::autograd::AutogradContext *ctx, 
    torch::autograd::tensor_list grad_outputs
)
{
    auto dOutput = grad_outputs[0];
    auto forward_pass_vars = ctx->get_saved_variables();
    auto X = forward_pass_vars[0];
    auto W = forward_pass_vars[1];
    auto b = forward_pass_vars[2];

    // dX = dOutput * W^T
    // dW = X^T * dOutput
    // db = sum(dOutput) across dim 0

    auto dX = torch::empty_like(X);
    auto dW = torch::empty_like(W);
    auto db = torch::empty_like(b);

    gemm_2d(dOutput, W.transpose(0,1).contiguous(), dX);
    gemm_2d(X.transpose(0,1).contiguous(), dOutput, dW);
    db = dOutput.sum(0);
    return {dX, dW, db};
}

// Activation Functions