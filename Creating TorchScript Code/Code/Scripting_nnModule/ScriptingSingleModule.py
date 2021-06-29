# Scripting a Single Module
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        
        # This parameter will be copied to the new ScriptModule
        self.weight = nn.Parameter(torch.randn(N, M))

        # When the submodule is used, it will be compiled
        self.linear = nn.Linear(N, M)
    
    def forward(self, x):
        output = self.weight.mv(x)

        # This calls the `forward` method of `nn.Linear` module, which will
        # cause the `self.linear` submodule to be compiled to a `ScriptModule`
        output = self.linear(output)
        return output

scripted_module = torch.jit.script(MyModule(2, 3)) 
print(scripted_module)