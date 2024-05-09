import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
        pid,
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def fused_add_mul_relu_kernel(input_ptr,
                              weights_ptr1,
                              bias_ptr1,
                              weights_ptr2,
                              bias_ptr2,
                              output_ptr,
                              batch_size,  # or stride along the batch dimension
                              num_features,  # or stride along the feature dimension
                              num_weights1,  # or stride along the weight dimension for linear layer 1
                              num_weights2,  # or stride along the weight dimension for linear layer 2
                              BLOCK_SIZE_B: tl.constexpr,
                              BLOCK_SIZE_F: tl.constexpr,
                              BLOCK_SIZE_W1: tl.constexpr,
                              BLOCK_SIZE_W2: tl.constexpr,
                              GROUP_SIZE: tl.constexpr):
    # Calculate the INPUT @ W1
    # Add Bias
    # relu
    # Calculate the result @ W2
    # Add Bias
    # sigmoid
    pid_w1 = tl.program_id(axis=0)
    pid_w2 = tl.program_id(axis=0)
    intermediate_result_input_w1 = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_W1), dtype=tl.float32)
    matmul_kernel(pid_w1, input_ptr, weights_ptr1, intermediate_result_input_w1,
                  batch_size, num_weights1, num_features,
                  num_features, 1,
                  num_weights1, 1,
                  num_weights1, 1,
                  BLOCK_SIZE_B, BLOCK_SIZE_W1, BLOCK_SIZE_F, GROUP_SIZE)
    # Add Bias
    offsets = pid_w1 * BLOCK_SIZE_W1 + tl.arange(0, BLOCK_SIZE_W1)
    mask = offsets < (num_weights1 * batch_size)
    bias_index = offsets % num_weights1
    bias_chunk = tl.load(bias_ptr1 + bias_index, mask, eviction_policy='evict_last')
    intermediate_result_input_w1 += bias_chunk

    # Apply ReLU
    intermediate_result_input_w1 = tl.maximum(0, intermediate_result_input_w1)

    # Calculate the result @ W2
    intermediate_result_w2 = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_W2), dtype=tl.float32)
    matmul_kernel(pid_w2, intermediate_result_input_w1, weights_ptr2, intermediate_result_w2,
                  batch_size, num_weights2, num_weights1,
                  num_weights1, 1,
                  num_weights2, 1,
                  num_weights2, 1,
                  BLOCK_SIZE_B, BLOCK_SIZE_W2, BLOCK_SIZE_W1, GROUP_SIZE)
    # Add Bias
    offsets = pid_w2 * BLOCK_SIZE_W2 + tl.arange(0, BLOCK_SIZE_W2)
    mask = offsets < (num_weights2 * batch_size)
    bias_index = offsets % num_weights2
    bias_chunk = tl.load(bias_ptr2 + bias_index, mask, eviction_policy='evict_last')
    intermediate_result_w2 += bias_chunk

    # Apply Sigmoid
    intermediate_result_w2 = tl.sigmoid(intermediate_result_w2)
    tl.store(output_ptr + offsets, intermediate_result_w2, mask=mask)


def fused_add_relu(input_tensor: torch.Tensor, weights1: torch.Tensor, bias1: torch.Tensor,
                   weights2: torch.Tensor, bias2: torch.Tensor) -> torch.Tensor:
    # print("calling fused_add_mul_relu_torch")
    grid = lambda meta: (triton.cdiv(input_tensor.size()[0], meta['BLOCK_SIZE_B']) * triton.cdiv(weights2.size()[1], meta["BLOCK_SIZE_W2"]),)
    BLOCK_SIZE_B = min(1024, input_tensor.numel())
    BLOCK_SIZE_F = min(1024, input_tensor.size(1))
    BLOCK_SIZE_W1 = min(1024, weights1.size(1))
    BLOCK_SIZE_W2 = min(1024, weights2.size(1))
    GROUP_SIZE = 1
    output_tensor = torch.zeros((input_tensor.size(0), weights2.size(1)),
                                device=input_tensor.device,
                                dtype=torch.float32)
    fused_add_mul_relu_kernel[grid](input_tensor, weights1, bias1, weights2, bias2, output_tensor,
                                    input_tensor.size(0), input_tensor.size(1), weights1.size(1), weights2.size(1),
                                    BLOCK_SIZE_B, BLOCK_SIZE_F, BLOCK_SIZE_W1, BLOCK_SIZE_W2, GROUP_SIZE)
    return output_tensor


if __name__ == '__main__':
    torch.cuda.set_device(0)  # no-op to ensure context
    input_tensor = torch.ones(size=(128, 512), device='cuda')
    weights1 = torch.ones(size=(512, 256), device='cuda')
    bias1 = torch.ones(size=(256,), device='cuda')
    weights2 = torch.ones(size=(256, 128), device='cuda')
    bias2 = torch.ones(size=(128,), device='cuda')
    eager_result = torch.sigmoid(torch.relu(input_tensor @ weights1 + bias1) @ weights2 + bias2)
    print(eager_result[:3, :3])
    triton_result = fused_add_relu(input_tensor, weights1, bias1, weights2, bias2)
    print(triton_result[:3, :3])
