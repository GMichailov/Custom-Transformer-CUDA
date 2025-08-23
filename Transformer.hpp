#pragma once

#include <torch/torch.h>
#include <torch/autograd.h>
#include <cublas_v2.h>

extern cublasHandle_t handle;

// Positional Encoders

struct PatchPositionalEmbedding : public torch::autograd::Function<PatchPositionalEmbedding> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor PositionalEmbeddings
    );
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
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
        torch::Tensor Wo,
        int num_heads
    );
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, 
        torch::autograd::tensor_list grad_outputs
    );
};

// MLP

struct Linear : public torch::autograd::Function<Linear> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor b
    );
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx, 
        torch::autograd::tensor_list grad_outputs
    );
};


