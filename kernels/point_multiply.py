import triton
import triton.language as tl
import torch


#Accepts mx1 vectors
@triton.jit
def pointmultiply_kernel(a_ptr, b_ptr, c_ptr, stride_a, stride_b, stride_c, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M

    a = tl.load(a_ptr + offs * stride_a, mask=mask)
    b = tl.load(b_ptr + offs * stride_b, mask=mask)

    c = a * b

    tl.store(c_ptr + offs * stride_c, c, mask=mask)


def point_multiply(a: torch.tensor, b: torch.tensor):
    if a.shape != b.shape:
        raise Exception("Attempting to point multiply tensors which are not the same shape")

    # Move CPU tensors to GPU
    a = a.to("cuda")
    b = b.to("cuda")

    BLOCK_SIZE = 32

    M = a.shape[0]

    c = torch.empty(M, dtype=torch.float32, device='cuda')

    # Strides
    stride_a = a.stride()[0]
    stride_b = b.stride()[0]
    stride_c = c.stride()[0]


    # Launch the kernel with a 2D grid: (M, N) blocks, each computing one element
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE),)

    pointmultiply_kernel[grid](a, b, c, stride_a, stride_b, stride_c, M, BLOCK_SIZE=BLOCK_SIZE)

    return c.cpu()



if __name__ == "__main__":
    M = 20
    a = torch.rand((M), dtype=torch.float32, device="cuda")
    b = torch.rand((M), dtype=torch.float32, device="cuda")
    
    c = pointmultiply(a, b)

    print("A: ", a)
    print()
    print("B: ", b)
    print()
    print(c)