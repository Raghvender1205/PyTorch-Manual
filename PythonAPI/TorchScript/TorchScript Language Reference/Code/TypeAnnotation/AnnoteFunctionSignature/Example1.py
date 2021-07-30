# Using Python3 style.
import torch

def f(a, b: int):
    return a + b

m = torch.jit.script(f)
print('TorchScript: ', m(torch.ones([6]), 100))