import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, TILE_SIZE_M: tl.constexpr, TILE_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
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
    tl.store(c_ptrs, c_tile, mask=c_mask)

def matmul(a: torch.tensor, b: torch.tensor):
    if a.shape[1] != b.shape[0]:
        raise Exception("Matrices have incompatible dimensions")

    # Move CPU tensors to GPU
    a = a.to("cuda")
    b = b.to("cuda")

    TILE_SIZE_M = 32
    TILE_SIZE_N = 32
    BLOCK_SIZE_K = 32

    M, K = a.shape
    _, N = b.shape

    c = torch.empty(M, N, dtype=torch.float32, device='cuda')

    # Strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Launch the kernel with a 2D grid: (M, N) blocks, each computing one element
    grid = lambda META: (triton.cdiv(M, TILE_SIZE_M), triton.cdiv(N, TILE_SIZE_N))

    matmul_kernel[grid](
        a, b, c, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        TILE_SIZE_M=TILE_SIZE_M,
        TILE_SIZE_N=TILE_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K 
    )

    return c.cpu()



if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024  # Example sizes
    
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    C = matmul(A, B)

    #Verify result
    torch_result = A @ B
    print("Results Match: ", torch.allclose(C, torch_result, atol=1e-3))
    print("Torch Result: ", torch_result[:2][:2])
    print("C: ", C[:2][:2])