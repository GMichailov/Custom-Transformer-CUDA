#include "Autograd.hpp"
#include <torch/torch.h>

using namespace ember::autograd;

torch::Tensor training::memoryAware::MultiHeadAttention::forward(
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

    
}