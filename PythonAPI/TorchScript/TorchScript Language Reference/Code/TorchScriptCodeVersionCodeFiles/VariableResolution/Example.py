import torch
    
@torch.jit.script
def foo(x):
    if x < 0:
        y = 4
    print(y)