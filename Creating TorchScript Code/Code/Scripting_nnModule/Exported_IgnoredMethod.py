# An Exported and ignored method in a Module
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    @torch.jit.export
    def some_entry_point(self, input):
        return input + 10
    
    @torch.jit.ignore
    def python_only_fn(self, input):
        # This Function won't be compiled
        import pdb
        pdb.set_trace()
    
    def forward(self, input):
        if self.training:
            self.python_only_fn(input)
        return input * 99
    
scripted_module = torch.jit.script(MyModule())
print(scripted_module.some_entry_point(torch.randn(2, 2)))
print(scripted_module(torch.randn(2, 2)))