# Example 1

import torch

@torch.jit.script
class MyClass:
    def __init__(self, x: int):
        self.x = x

    def inc(self, val: int):
        self.x += val
