import triton
import triton.language as tl
import torch
@triton.jit
def add_kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr):
  start = BLOCK_SIZE * tl.program_id(axis=0)
  offsets =  start + tl.arange(0,BLOCK_SIZE)
  mask = offsets < n
  xs=tl.load(x+offsets, mask=mask)
  ys=tl.load(y+offsets, mask=mask)
  tl.store(out+offsets, xs+ys, mask=mask)

def add(x, y):
  out = torch.empty_like(x).cuda()
  grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
  add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
  return out

@triton.testing.perf_report(
    triton.testing.Benchmark(
      x_names=['size'],
      x_vals=[2**i for i in range(12,28,1)],
      x_log=True,
      line_arg='provider',
      line_vals=['triton', 'torch'],
      line_names=['triton', 'torch'],
      styles=[('blue', '-'), ('green', '-')],
      ylabel='GB/s',
      plot_name='vector-add-performance',
      args={}
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True,show_plots=True)

