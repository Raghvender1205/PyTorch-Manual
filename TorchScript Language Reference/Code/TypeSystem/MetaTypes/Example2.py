import torch
from typing import Any

def f(a: Any):
    print(a)
    return (isinstance(a, torch.Tensor))

ones = torch.ones([2])
m = torch.jit.script(f)
print(m(ones))