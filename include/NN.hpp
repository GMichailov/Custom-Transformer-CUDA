#pragma once
#include "Operations.hpp"


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