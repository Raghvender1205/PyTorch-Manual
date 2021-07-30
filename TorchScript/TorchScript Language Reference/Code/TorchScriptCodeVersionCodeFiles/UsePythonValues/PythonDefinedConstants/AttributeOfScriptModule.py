import torch
import torch.nn as nn

class Foo(nn.Module):
    # `Final` from the `typing_Extensions` module can also be used.
    a: torch.jit.Final[int]

    def __init__(self):
        super(Foo, self).__init__()
        self.a = 5
    
    def forward(self, x):
        return self.a + x
    
f = torch.jit.script(Foo())
print(f)