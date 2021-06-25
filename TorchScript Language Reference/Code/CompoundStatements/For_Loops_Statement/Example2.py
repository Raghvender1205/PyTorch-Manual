# For loops on lists; for loops over a torch.nn.ModuleList will unroll the body of the loop at compile time, with each member of the module list.

import torch

class SubModule(torch.nn.ModuleList):
    def __init__(self):
        super(SubModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(2))
    
    def forward(self, input):
        return self.weight + input
    
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])
    
    def forward(self, v):
        for module in self.mods:
            v = module(v)
        return v

model = torch.jit.script(MyModule())

print(model)