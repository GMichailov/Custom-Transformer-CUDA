import torch
from torch.autograd import gradcheck

dtype = torch.double

# === Hardcoded tensors from your C++ run ===
X = torch.tensor([[[ 0.5771,  0.0916,  0.6537, -0.7872],
                   [-1.4698, -1.6744, -0.6908, -0.9245],
                   [ 0.7359,  1.4721, -0.1543, -0.8554]],
                  [[-0.2452,  0.5674, -0.2925,  0.3129],
                   [-0.5934, -0.8286, -0.6482, -0.7700],
                   [ 1.5485, -0.0816, -0.8945, -2.8470]]],
                 dtype=dtype, requires_grad=True)

W = torch.tensor([[ 0.5237, -1.3734,  0.9406,  0.3202,  0.2472],
                  [-0.0752, -0.8124, -1.1429,  1.4445, -0.2364],
                  [ 0.1707, -0.9478,  0.8407,  2.9200,  0.1810],
                  [ 1.3269,  0.1174, -0.0761,  0.6502, -0.8246]],
                 dtype=dtype, requires_grad=True)

b = torch.tensor([-0.6849, -1.0209,  0.8500,  0.3319,  0.4889],
                 dtype=dtype, requires_grad=True)

# === Reference Linear ===
def linear_ref(X, W, b):
    return X.matmul(W) + b

# === Forward ===
output = linear_ref(X, W, b)
print("Forward output (PyTorch):\n", output)

# === Expected forward (from your C++ run) ===
print("\nExpected Forward (C++ run):")
print("""(1,.,.) = 
-0.6849 -1.0209  0.8500  0.3319  0.4889
-0.6849 -1.0209  0.8500  0.3319  0.4889
-0.6849 -1.0209  0.8500  0.3319  0.4889

(2,.,.) = 
-0.6849 -1.0209  0.8500  0.3319  0.4889
-0.6849 -1.0209  0.8500  0.3319  0.4889
-0.6849 -1.0209  0.8500  0.3319  0.4889
[ CPUFloatType{2,3,5} ]""")

# === Backward (with dOutput = ones) ===
grad_output = torch.ones_like(output)
output.backward(grad_output)

print("\n--- Gradients (PyTorch) ---")
print("Grad X:\n", X.grad)
print("Grad W:\n", W.grad)
print("Grad b:\n", b.grad)

# === Expected gradients (from your C++ run) ===
print("\n--- Expected Gradients (C++ run) ---")
print("""Grad X:
(1,.,.) = 
 0.6583 -0.8225  3.1646  1.1938
 0.6583 -0.8225  3.1646  1.1938
 0.6583 -0.8225  3.1646  1.1938

(2,.,.) = 
 0.6583 -0.8225  3.1646  1.1938
 0.6583 -0.8225  3.1646  1.1938
 0.6583 -0.8225  3.1646  1.1938
[ CPUFloatType{2,3,4} ]

Grad W:
 0.5531  0.5531  0.5531  0.5531  0.5531
-0.4534 -0.4534 -0.4534 -0.4534 -0.4534
-2.0265 -2.0265 -2.0265 -2.0265 -2.0265
-5.8714 -5.8714 -5.8714 -5.8714 -5.8714
[ CPUFloatType{4,5} ]

Grad b:
 6
 6
 6
 6
 6
[ CPUFloatType{5} ]""")

# === Gradcheck ===
X.grad.zero_(); W.grad.zero_(); b.grad.zero_()
inputs = (X, W, b)
print("\nGradcheck result:", gradcheck(linear_ref, inputs, eps=1e-6, atol=1e-4))
