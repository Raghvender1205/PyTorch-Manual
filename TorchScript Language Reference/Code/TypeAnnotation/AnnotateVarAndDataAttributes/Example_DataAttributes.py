import torch

class MyModule(torch.nn.Module):
    offset_: int

def __init__(self, offset):
    self.offset_ = offset

...