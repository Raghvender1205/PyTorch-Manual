import torch

@torch.jit.script
class MyClass(object):
    name = "MyClass"
    def __init__(self, x: int):
        self.x = x

def fn(a: MyClass):
    return a.name