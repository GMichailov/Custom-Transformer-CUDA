#pragma once
#include "Operations.hpp"


namespace ember {
namespace autograd {

namespace inference {
namespace memoryAware {

/*
 * Will later be implemented similar to vllm, but to allow users to run proportionally large models compared to the amount of available RAM. (CPU Implementation for Raspberry Pi?)
 */
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

}
namespace speed {

/*
 * Optimized for inference under the assumption that it runs with a single input, cannot handle batches.
 */
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

}
}

namespace training {
namespace memoryAware {

/*
 * Different variants store less and less for the backward pass and rely more and more heavily on recomputation.
 */
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

}
namespace speed {

/*
 * Different variants store more and more for the backward pass and rely more and more heavily on memory.
 */
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

}
}

}
}