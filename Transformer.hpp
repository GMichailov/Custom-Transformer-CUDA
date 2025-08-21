#pragma once

#include <torch/torch.h>
#include <torch/autograd.h>
#include <cublas_v2.h>

extern cublasHandle_t handle;

// Calculation Functions

void single_attention_matrix_mul_gemm(
    const torch::Tensor& src_a,
    const torch::Tensor& src_b,
    torch::Tensor& dest
);

// Positional Encoders

struct AbsolutePositionalEncoder : public torch::autograd::Function<AbsolutePositionalEncoder> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx
    );
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx
    );
};

// Attention

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

struct MultiHeadAttention : public torch::autograd::Function<MultiHeadAttention> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Wq,
        torch::Tensor Wk,
        torch::Tensor Wv,
        int num_heads
    );
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, 
        torch::autograd::tensor_list grad_outputs
    );
};

// Activation Funtions

struct SwiGlu : public torch::autograd::Function<SwiGlu> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx
    );
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx
    );
};
