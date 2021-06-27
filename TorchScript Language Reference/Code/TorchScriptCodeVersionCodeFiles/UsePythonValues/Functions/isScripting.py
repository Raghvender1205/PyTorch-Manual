import torch

@torch.jit.unused
def unsupported_linear_op(x):
    return x

def linear(x):
    if torch.jit.is_scripting():
        return torch.linear(x)
    else:
        return unsupported_linear_op(x)