#pragma once

#include <torch/torch.h>
#include <torch/autograd.h>
#include <cublas_v2.h>

extern cublasHandle_t handle;

void single_attention_matrix_mul_gemm(
    const torch::Tensor& src_a,
    const torch::Tensor& src_b,
    torch::Tensor& dest
);

struct SingleHeadAttention : public torch::autograd::Function<SingleHeadAttention> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Wq,
        torch::Tensor Wk,
        torch::Tensor Wv
    );
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, 
        torch::autograd::tensor_list grad_outputs
    );
};
