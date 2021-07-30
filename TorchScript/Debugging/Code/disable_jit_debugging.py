# To run this script use the following command
# Create PYTORCH_JIT as an Environment Variable and set it to 0, for Disabling JIT Debugging

# PYTORCH_JIT=0 python disable_jit_debugging.py

import torch
import pdb


@torch.jit.script
def scripted_fn(x: torch.Tensor):
    for i in range(12):
        x += x
    return x


def fn(x):
    x = torch.neg(x)
    pdb.set_trace()
    return scripted_fn(x)


traced_fn = torch.jit.trace(fn, (torch.rand(4, 5)))
traced_fn(torch.rand(3, 4))
