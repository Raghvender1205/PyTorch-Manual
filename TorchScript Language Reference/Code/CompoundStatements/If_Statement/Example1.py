# A tensor with 1 dimension is promoted to bool:

import torch

@torch.jit.script
def fn(x: torch.Tensor):
    if x:  # If Tensor is promoted
        return True
    return False

print(fn(torch.rand(1)))
