import torch


@torch.jit.script
def error(x):
    if x:
        r = torch.rand(1)
    else:
        r = 4
    return r

