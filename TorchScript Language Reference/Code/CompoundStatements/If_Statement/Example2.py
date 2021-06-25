# A tensor with multi dimensions are not promoted to bool

import torch
# Multi dimensional Tensors are error out

@torch.jit.script
def fn():
    if torch.rand(2):
        print('Tensor is available')

    if torch.rand(4, 5, 6):
        print('Tensor is available')

print(fn())