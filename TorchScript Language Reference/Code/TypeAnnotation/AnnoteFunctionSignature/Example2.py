# MyPy Style 
import torch 

def f(a, b):
    # type: (torch.Tensor, int) -> torch.Tensor
    return a + b

m = torch.jit.script(f)
print('TorchScript: ', m(torch.ones([6]), 100))