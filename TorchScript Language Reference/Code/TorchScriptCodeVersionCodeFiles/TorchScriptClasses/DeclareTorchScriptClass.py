# After a class is defined, it can be used in both TorchScript and Python interchangeably 
# like any other TorchScript type:
import torch

@torch.jit.script
class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
@torch.jit.script
def sum_pair(p):
    # type: (Pair) -> Tensor
    return p.first + p.second

p = Pair(torch.rand(2, 3), torch.rand(2, 3))
print(sum_pair(p))