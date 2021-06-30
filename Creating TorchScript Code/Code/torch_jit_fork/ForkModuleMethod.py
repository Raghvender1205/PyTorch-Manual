import torch
from torch import Tensor
class AddMod(torch.nn.Module):
    def forward(self, a: Tensor, b : int):
        return a + b
    
class Mod(torch.nn.Module):
    def __init__(self):
        super(self).__init__()
        self.mod = AddMod()
    def forward(self, input):
        fut = torch.jit.fork(self.mod, a, b=2)
        return torch.jit.wait(fut)

input = torch.tensor(2)
mod = Mod()
assert mod(input) == torch.jit.script(mod).forward(input)