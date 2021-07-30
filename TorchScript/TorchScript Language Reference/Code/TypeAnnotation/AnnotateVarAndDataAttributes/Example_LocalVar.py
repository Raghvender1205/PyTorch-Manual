import torch
from typing import Optional

def f(a, setVal: bool):
    value: Optional[torch.Tensor] = None
    if setVal:
        value = a
    return value

ones = torch.ones([6])
m = torch.jit.script(f)
print('TorchScript: ', m(ones, True), m(ones, False))