import torch
from torch.autograd import gradcheck

# Fixed seed for reproducibility
torch.manual_seed(0)

# === Your test tensors (same values as C++ run) ===
X = torch.tensor([[[0.5130,  1.0139,  0.1489],
                   [0.7487,  0.0688, -1.9271]]],
                 dtype=torch.double, requires_grad=True)

Wq = torch.tensor([[ 1.3489,  0.1496, -1.0931],
                   [-0.2162, -1.8937,  1.9240],
                   [-1.2784, -0.1501, -0.3060]],
                  dtype=torch.double, requires_grad=True)

Wk = torch.tensor([[-0.2472,  1.4384, -0.3051],
                   [-1.2521,  0.2389, -0.1467],
                   [-1.3271, -1.1168, -0.9000]],
                  dtype=torch.double, requires_grad=True)

Wv = torch.tensor([[ 0.3242, -1.1716,  0.1092],
                   [-0.4310, -0.9350, -0.8025],
                   [ 0.4390,  0.5588, -0.1643]],
                  dtype=torch.double, requires_grad=True)

# === Define reference attention using PyTorch autograd ===
def single_head_attention(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = (Q @ K.transpose(-2, -1)) / (Wq.size(1) ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

# === Wrap in gradcheck function ===
def func(*inputs):
    return single_head_attention(*inputs)

# Run gradcheck
inputs = (X, Wq, Wk, Wv)
print("Gradcheck result:", gradcheck(func, inputs, eps=1e-6, atol=1e-4))