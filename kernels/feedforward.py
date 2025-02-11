import triton
import triton.language as tl
import torch

from kernels.activations import leaky_relu

@triton.jit
def feedforward_kernel(a_ptr, b_ptr, c_ptr, preact_ptr, bias_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_pm, stride_pn, stride_bias, ACTIVATION: tl.constexpr, TILE_SIZE_M: tl.constexpr, TILE_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rows = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    cols = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)

    c_tile = tl.zeros((TILE_SIZE_M, TILE_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        vals = k + tl.arange(0, BLOCK_SIZE_K)

        a_mask = vals[None,:] < K
        b_mask = vals[:, None] < K

        a_ptrs = a_ptr + (rows[:, None] * stride_am) + (vals[None, :] * stride_ak)
        b_ptrs = b_ptr + (vals[:, None] * stride_bk) + (cols[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=a_mask)
        b = tl.load(b_ptrs, mask=b_mask)

        c_tile += tl.dot(a, b)
    
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    c_ptrs = c_ptr + (rows[:, None] * stride_cm) + (cols[None, :] * stride_cn)
    preact_ptrs = preact_ptr + (rows[:, None] * stride_pm) + (cols[None, :] * stride_pn)

    #Add Bias
    bias_offs = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)
    bias_mask = bias_offs < N

    bias = tl.load(bias_ptr + bias_offs * stride_bias, mask=bias_mask)

    c_tile = c_tile + bias[None, :]

    tl.store(preact_ptrs, c_tile, mask=c_mask)

    if ACTIVATION == "leaky_relu":
        c_tile = leaky_relu(c_tile)

    tl.store(c_ptrs, c_tile, mask=c_mask)


def feedforward_op(inputs: torch.tensor, weights_transposed: torch.tensor, bias: torch.tensor):
    # Move CPU tensors to GPU
    inputs = inputs.to("cuda")
    weights_transposed = weights_transposed.to("cuda")
    bias = bias.to("cuda")

    TILE_SIZE_M = 32
    TILE_SIZE_N = 32
    BLOCK_SIZE_K = 32

    M, K = inputs.shape
    _, N = weights_transposed.shape

    activations = torch.empty(M, N, dtype=torch.float32, device='cuda')
    preactivations = torch.empty(M, N, dtype=torch.float32, device='cuda')

    # Strides
    stride_am, stride_ak = inputs.stride()
    stride_bk, stride_bn = weights_transposed.stride()
    stride_cm, stride_cn = activations.stride()
    stride_pm, stride_pn = preactivations.stride()
    stride_b = bias.stride()[0]

    # Launch the kernel with a 2D grid: (M, N) blocks, each computing one element
    grid = lambda META: (triton.cdiv(M, TILE_SIZE_M), triton.cdiv(N, TILE_SIZE_N))

    feedforward_kernel[grid](
        inputs, weights_transposed, activations, preactivations, bias, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_pm, stride_pn,
        stride_b,
        ACTIVATION = "leaky_relu",
        TILE_SIZE_M=TILE_SIZE_M,
        TILE_SIZE_N=TILE_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K 
    )

    return preactivations.cpu(), activations.cpu()

if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024  # Example sizes
    TILE_SIZE_M = 64
    TILE_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    C = torch.empty(M, N, dtype=torch.float32, device='cuda')
    bias = torch.randn(2048, 1, dtype=torch.float32, device='cuda') * 10

    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    stride_b, _ = bias.stride()

    # Launch the kernel with a 2D grid: (M, N) blocks, each computing one element
    grid = lambda META: (triton.cdiv(M, TILE_SIZE_M), triton.cdiv(N, TILE_SIZE_N))

    feedforward_kernel[grid](
        A, B, C, bias, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_b,
        ACTIVATION = "leaky_relu",
        TILE_SIZE_M=16,
        TILE_SIZE_N=16,
        BLOCK_SIZE_K=16
    )

    #Verify result
    torch_result = A @ B
    print("Results Match: ", torch.allclose(C, torch_result, atol=1e-3))
    print("Torch Result: ", torch_result[:2][:2])
    print("C: ", C[:2][:2])