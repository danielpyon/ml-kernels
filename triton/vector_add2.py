import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  start = pid * BLOCK_SIZE

  offsets = tl.arange(0, BLOCK_SIZE)
  xs = tl.load(x_ptr + start + offsets)
  ys = tl.load(y_ptr + start + offsets)

  tl.store(output_ptr + start + offsets, xs + ys)

def rel_error(x, y):
  scale = max(x.shape[0], y.shape[0])
  return (torch.abs(x-y) / scale).sum().item()

if __name__ == '__main__':
  n_elements = 64
  x = torch.randn(n_elements).cuda()
  y = torch.randn(n_elements).cuda()
  z = x + y
  output = torch.empty_like(x).cuda()

  BLOCK_SIZE = 16
  add_kernel[(4,)](x, y, output, n_elements, BLOCK_SIZE)

  assert rel_error(z, output) <= 1e-7
