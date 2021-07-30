import torch
import torch.nn as nn

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))

    def forward(self, input):
        return self.weight + input

class MyModule(nn.Module):
    __constants__ = ['mods']

    def __init__(self):
        super(MyModule, self).__init__()
        self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])
    
    def forward(self, v):
        for module in self.mods:
            v = module(v)
        return v

m = torch.jit.script(MyModule())
print(m)