import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  start = pid * BLOCK_SIZE

  # need to mask out stuff
  offsets = start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements

  # tl.load takes a list of pointers
  xs = tl.load(x_ptr + offsets, mask=mask)
  ys = tl.load(y_ptr + offsets, mask=mask)

  # store takes a list of pointers and some values
  tl.store(output_ptr + offsets, xs + ys, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  assert x.shape == y.shape
  n_elements = x.numel()

  out = torch.empty_like(x)

  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

  return out


def rel_error(x, y):
  scale = max(x.shape[0], y.shape[0])
  return (torch.abs(x-y) / scale).sum().item()

if __name__ == '__main__':
  n_elements = 123456789
  x = torch.randn(n_elements).cuda()
  y = torch.randn(n_elements).cuda()
  out = add(x, y)
  z = x + y

  assert rel_error(out, z) <= 1e-7
