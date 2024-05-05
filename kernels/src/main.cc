#include <iostream>
#include "pointwise_add_relu_fused.cuh"
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>


int main()
{
    std::vector<int64_t> sizes = {2, 3};
    auto x = torch::randn(sizes, torch::kCUDA);
    std::cout << "Tensor x:\n" << x << '\n';
    auto y = torch::randn(sizes, torch::kCUDA);
    std::cout << "Tensor y:\n" << y << '\n';
    auto expected_result = torch::clamp_min(x + y, 0.0);

    auto result = pointwise_add_relu_fusion_512(x, y);

    std::cout << "Actual:\n" << result << '\n';
    std::cout << "Expected:\n" << expected_result << '\n';

    return 0;
}
