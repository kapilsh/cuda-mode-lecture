#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "pointwise_add_relu_fused.cuh"


__global__
void pointwise_add_relu_fusion_512_kernel(float* in_out_ptr0, const float* in_ptr0, int xnumel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < xnumel) {
        float tmp0 = in_out_ptr0[i];
        float tmp1 = in_ptr0[i];
        float tmp3 = fmaxf(0.0f, tmp0 + tmp1);
        in_out_ptr0[i] = tmp3;
    }
}

torch::Tensor pointwise_add_relu_fusion_512(torch::Tensor in_out, const torch::Tensor& in) {
    int XBLOCK = 512;
    auto numel = in_out.numel();
    dim3 threadsPerBlock(XBLOCK);
    dim3 numBlocks((numel + XBLOCK - 1) / XBLOCK);
    pointwise_add_relu_fusion_512_kernel<<<numBlocks, threadsPerBlock>>>(in_out.data_ptr<float>(), in.data_ptr<float>(), numel);
    cudaDeviceSynchronize();
    return in_out;
}