# This Example shows an incorrect usage of module type. Specifically, this example invokes the constructor of TestModule inside the scope of TorchScript.
import torch

class TestModule(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.x = v
    
    def forward(self, x: int):
        return self.x + x

class MyModel:
    def __init__(self, v: int):
        self.val = v
    
    @torch.jit.export
    def do(self, val: int) -> int:
        # ERROR: should not invoke the constructor of module type
        model = TestModule(self.val)
        return model(val)

# m = torch.jit.script(MyModel(2)) # Results in below RuntimeError
# RuntimeError: Could not get name of python class object
