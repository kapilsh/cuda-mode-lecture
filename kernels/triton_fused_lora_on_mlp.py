import torch
import triton
import triton.language as tl


@triton.jit
def get_1d_index(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)


@triton.jit
def get_2d_index(offs_0, offs_1, stride_0, stride_1=1):
    return tl.expand_dims(offs_0, 1) * stride_0 + tl.expand_dims(offs_1, 0) * stride_1


@triton.jit
def get_1d_mask(offs, max):
    return offs < max


@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)


@triton.jit
def fused_add_mul_relu(input_ptr,
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
                       BLOCK_SIZE_W2: tl.constexpr):
    
    # Step 1: step1 = tl.dot(block_bn, block_nw1)
    # Step 2: block_1w1 = index_bn % W1.size()[1]
    # Step 3: step3 = step1 + block_1w1
    # Step 4: step4 = Relu(step3)
    # Step 5: block_w1w2 = ...
    # Step 6: step6 = tl.dot(step4, block_w1w2)
    # Step 7: block_1w2 = index_w1w2 % W2.size()[1]
    # Step 8: step8 = step6 + block_1w2
    # Step 9: result = sigmoid(step8)
    
    pid_io = tl.program_id(0)
    pid_w1 = tl.program_id(1)
    pid_w2 = tl.program_id(2)

    # index along the rows for the input/output tensor
    input_start_index = get_1d_index(BLOCK_SIZE_B, pid_io)
    # index along the rows for W1
    w1_start_index = get_1d_index(BLOCK_SIZE_F, pid_w1)
    # index along the rows for W2
    w2_start_index = get_1d_index(BLOCK_SIZE_W1, pid_w2)

    # 2d block index for the input/output tensor
    input_index = input_ptr + get_2d_index(input_start_index, w1_start_index, batch_size, num_features)
    # 2d block index for W1 - to multiply with the input/output tensor block
    w1_index = weights_ptr1 + get_2d_index(w1_start_index, w2_start_index, num_features, num_weights1)
    # 2d block index for W2 - to multiply with the result of the multiplication of W1 and the input/output tensor
    w2_index = weights_ptr2 + get_2d_index(w2_start_index, 0, num_weights1, num_weights2)

    intermediate_result_iow1 = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_W1), dtype=tl.float32)
    for _ in range(0, num_features, BLOCK_SIZE_F):
        # ignore the masks for now
        input_block = tl.load(input_index, None)
        w1_block = tl.load(w1_index, None)
        intermediate_result_iow1 += tl.dot(input_block, w1_block, allow_tf32=False)
        input_index += BLOCK_SIZE_F * num_features
        w1_index += BLOCK_SIZE_F * num_features

    # add bias
    output_index = output_ptr + get_2d_index(input_start_index, w2_start_index, )
