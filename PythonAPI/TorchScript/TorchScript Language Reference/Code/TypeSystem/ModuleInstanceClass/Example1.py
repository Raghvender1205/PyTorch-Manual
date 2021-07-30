# This Example illustrates a few features of module types
import torch

class TestModule(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.x = v
    
    def forward(self, inc: int):
        return self.x + inc


m = torch.jit.script(TestModule(1))
print(f'First Instance : {m(3)}')

m = torch.jit.script(TestModule(torch.ones([5])))
print(f"Second Instance: {m(3)}")