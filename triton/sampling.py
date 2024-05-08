import triton

@triton.jit
def sample_from_prob_kernel(x_ptr,  # *Pointer* to first input vector.
                            output_ptr,  # *Pointer* to output vector.
                            n_elements,  # Size of the vector.
                            BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                            ):
  
  # There are multiple 'programs' processing different data. We identify which program
  # we are here:
  pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

  # This program will process inputs that are offset from the initial data.
  # For instance, if you had a vector of length 256 and block_size of 64, the programs
  # would each access the elements [0:64, 64:128, 128:192, 192:256].
  # Note that offsets is a list of pointers:
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  # Create a mask to guard memory operations against out-of-bounds accesses.
  mask = offsets < n_elements

  # Load x and y from DRAM, masking out any extra elements in case the input is not a
  # multiple of the block size.
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x*2

  # Write x + y back to DRAM.
  tl.store(output_ptr + offsets, output, mask=mask)

def sample_from_prob():
  # We need to preallocate the output.
  output = torch.empty_like(x)
  assert x.is_cuda and y.is_cuda and output.is_cuda
  n_elements = output.numel()
  # The SPMD launch grid denotes the number of kernel instances that run in parallel.
  # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
  # In this case, we use a 1D grid where the size is the number of blocks:
  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
  # NOTE:
  #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
  #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
  #  - Don't forget to pass meta-parameters as keywords arguments.
  sample_from_prob_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
  # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
  # running asynchronously at this point.
  return output

if __name__ == '__main__':
  sample_from_prob()
