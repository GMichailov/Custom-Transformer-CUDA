#include "OriginalKernels.hpp"
#include <iostream>
#include <torch/autograd.h>


void logging_gradcheck() {
    std::cout << "Creating X and weight matrices.\n";

    auto X = torch::randn({1, 2, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
    auto Wq = torch::randn({3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
    auto Wk = torch::randn({3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
    auto Wv = torch::randn({3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));

    std::cout << "Input X:\n" << X.cpu() << std::endl;
    std::cout << "Weight Q:\n" << Wq.cpu() << std::endl;
    std::cout << "Weight K:\n" << Wk.cpu() << std::endl;
    std::cout << "Weight V:\n" << Wv.cpu() << std::endl;

    auto output = SingleHeadAttention::apply(X, Wq, Wk, Wv);

    std::cout << "Output:\n" << output.cpu() <<std::endl;

    output.backward(torch::ones_like(output));

    std::cout << "Grad X:\n" << X.grad().cpu() << std::endl;
    std::cout << "Grad Wq:\n" << Wq.grad().cpu() << std::endl;
    std::cout << "Grad Wk:\n" << Wk.grad().cpu() << std::endl;
    std::cout << "Grad Wv:\n" << Wv.grad().cpu() << std::endl;
}

void logging_gradcheck_multihead() {
    int batch_size = 1;
    int seq_len = 2;
    int d_model = 4;   // must be divisible by num_heads
    int num_heads = 2;
    int head_dim = d_model / num_heads;

    std::cout << "Creating X and weight matrices for MultiHeadAttention.\n";

    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true);

    auto X  = torch::randn({batch_size, seq_len, d_model}, opts);
    auto Wq = torch::randn({d_model, d_model}, opts);
    auto Wk = torch::randn({d_model, d_model}, opts);
    auto Wv = torch::randn({d_model, d_model}, opts);
    auto Wo = torch::randn({d_model, d_model}, opts);

    std::cout << "Input X:\n" << X.cpu() << "\n";
    std::cout << "Wq:\n" << Wq.cpu() << "\n";
    std::cout << "Wk:\n" << Wk.cpu() << "\n";
    std::cout << "Wv:\n" << Wv.cpu() << "\n";
    std::cout << "Wo:\n" << Wo.cpu() << "\n";

    // Forward
    auto output = MultiHeadAttention::apply(X, Wq, Wk, Wv, Wo, num_heads);
    std::cout << "Final Output (after Wo):\n" << output.cpu() << "\n";

    // Backward
    output.backward(torch::ones_like(output));

    // Print grads
    std::cout << "\n--- Gradients ---\n";
    std::cout << "Grad X:\n"  << X.grad().cpu()  << "\n";
    std::cout << "Grad Wq:\n" << Wq.grad().cpu() << "\n";
    std::cout << "Grad Wk:\n" << Wk.grad().cpu() << "\n";
    std::cout << "Grad Wv:\n" << Wv.grad().cpu() << "\n";
    std::cout << "Grad Wo:\n" << Wo.grad().cpu() << "\n";
}

void logging_gradcheck_linear() {
    int batch_size = 2;
    int seq_len    = 3;
    int d_in       = 4;
    int d_out      = 5;

    std::cout << "Creating X, W, b for Linear layer.\n";

    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true);

    auto X = torch::randn({batch_size, seq_len, d_in}, opts);
    auto W = torch::randn({d_in, d_out}, opts);
    auto b = torch::randn({d_out}, opts);

    std::cout << "Input X:\n" << X.cpu() << "\n";
    std::cout << "Weight W:\n" << W.cpu() << "\n";
    std::cout << "Bias b:\n" << b.cpu() << "\n";

    // Forward
    auto output = Linear::apply(X, W, b);
    std::cout << "Output:\n" << output.cpu() << "\n";

    // Backward
    output.backward(torch::ones_like(output));

    std::cout << "\n--- Gradients ---\n";
    std::cout << "Grad X:\n" << X.grad().cpu() << "\n";
    std::cout << "Grad W:\n" << W.grad().cpu() << "\n";
    std::cout << "Grad b:\n" << b.grad().cpu() << "\n";
}


int main() {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle!" << std::endl;
        return -1;
    }

    logging_gradcheck_linear();

    cublasDestroy(handle);
    return 0;
}