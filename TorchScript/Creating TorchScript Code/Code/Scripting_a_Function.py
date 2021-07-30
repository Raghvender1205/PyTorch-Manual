import torch

@torch.jit.script
def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r

print(type(foo)) # torch.jit.ScriptFunction

# Call the function using TorchScript interpreter
foo(torch.ones(2, 2), torch.ones(2, 2))