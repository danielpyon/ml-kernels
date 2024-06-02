import torch

import triton
import triton.language as tl

@torch.jit.script
def naive_softmax(x):
  m = torch.max(x)
  a = torch.exp(x - m)
  d = torch.sum(a)
  return a / d

print(naive_softmax([1,2,3,4]))

