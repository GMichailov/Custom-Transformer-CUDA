import torch
from torch.autograd import gradcheck

torch.manual_seed(0)

# === Hardcoded test values (copied from your C++ output) ===
X = torch.tensor([[[-0.9157, -1.0171,  1.2589,  0.6622],
                   [-1.6239,  0.6099, -0.3459,  0.2677]]],
                 dtype=torch.double, requires_grad=True)

Wq = torch.tensor([[ 0.7908,  0.0635,  1.7480, -1.6540],
                   [ 2.1136, -1.1247, -0.8736, -0.0695],
                   [ 0.1057,  0.4116,  0.1112,  1.7269],
                   [-0.3636, -0.5514, -1.2429, -0.1007]],
                  dtype=torch.double, requires_grad=True)

Wk = torch.tensor([[ 2.9617,  0.1250, -1.2411,  0.1634],
                   [ 1.2881,  1.2692, -1.7193, -0.2197],
                   [ 0.4274, -0.2026,  1.6830,  0.3585],
                   [-2.0125, -0.1445, -0.8421,  0.3113]],
                  dtype=torch.double, requires_grad=True)

Wv = torch.tensor([[ 1.2132,  0.8715,  0.0034,  0.6848],
                   [-0.3176,  2.0456, -0.2612,  0.9723],
                   [ 0.5331, -0.7213,  2.0104, -0.1108],
                   [-0.6709,  0.5882, -0.6255, -0.9776]],
                  dtype=torch.double, requires_grad=True)

Wo = torch.tensor([[-0.6463,  0.5629, -0.7447, -0.7374],
                   [-0.4864,  0.2037, -0.6853, -0.5386],
                   [-0.0244, -0.5147,  0.6585, -0.8844],
                   [ 0.4846, -1.1886,  1.3873, -0.0137]],
                  dtype=torch.double, requires_grad=True)

# === Multi-head attention function ===
def multihead_attention(X, Wq, Wk, Wv, Wo, num_heads=2):
    batch_size, seq_len, d_model = X.shape
    head_dim = d_model // num_heads

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Reshape into heads
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0,2,1,3)
    K = K.view(batch_size, seq_len, num_heads, head_dim).permute(0,2,1,3)
    V = V.view(batch_size, seq_len, num_heads, head_dim).permute(0,2,1,3)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)

    # Concatenate heads
    output = output.permute(0,2,1,3).contiguous().view(batch_size, seq_len, d_model)

    # Final output projection
    return output @ Wo

# === Forward check ===
out = multihead_attention(X, Wq, Wk, Wv, Wo, num_heads=2).detach()
print("Forward output:")
print(out)

# Expected from your C++ run
expected = torch.tensor([[[1.0405, 0.1395, 0.1382, 2.0160],
                          [1.6100, 0.3522, 0.8944, 3.0907]]],
                        dtype=torch.double)

print("\nExpected output:")
print(expected)

print("\nDifference:", (out - expected).abs().max().item())

# === Gradcheck ===
def func(*inputs):
    return multihead_attention(*inputs, num_heads=2)

inputs = (X, Wq, Wk, Wv, Wo)
print("\nGradcheck result:", gradcheck(func, inputs, eps=1e-6, atol=1e-4))