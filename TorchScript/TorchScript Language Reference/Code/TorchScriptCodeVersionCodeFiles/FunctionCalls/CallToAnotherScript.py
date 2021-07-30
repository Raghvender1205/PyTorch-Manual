import torch

@torch.jit.script
def foo(x):
    return x + 1

@torch.jit.script
def bar(x):
    return foo(x)
