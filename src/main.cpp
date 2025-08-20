#include "Transformer.hpp"
#include <iostream>

int main() {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle!" << std::endl;
        return -1;
    }

    std::cout << "Creating X and weight matrices.\n";

    auto X = torch::randn({2, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
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

    cublasDestroy(handle);
    return 0;
}