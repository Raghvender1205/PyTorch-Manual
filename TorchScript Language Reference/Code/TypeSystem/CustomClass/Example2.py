# Exampe 2 Custom Class
import torch

@torch.jit.script
class foo:
    def __init__(self):
        self.y = 1

# ERROR: self.x is not defined in __init__()
def assign_x(self):
    self.x = torch.rand(2, 3)