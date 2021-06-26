import torch

@torch.jit.script
class Foo:
    def __init__(self, x, y):
        self.x = x
    
    def aug_add_x(self, inc):
        self.x += inc

print("It works..!!")