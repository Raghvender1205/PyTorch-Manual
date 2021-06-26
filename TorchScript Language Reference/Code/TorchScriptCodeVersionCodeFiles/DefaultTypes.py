from typing import Tuple
import torch

@torch.jit.script
def foo(x, tup):
    # type: (int, Tuple[Tensor, Tensor]) -> Tensor
    t0, t1 = tup
    return t0 + t1 + x


print(foo(3, (torch.rand(3), torch.rand(3))))
