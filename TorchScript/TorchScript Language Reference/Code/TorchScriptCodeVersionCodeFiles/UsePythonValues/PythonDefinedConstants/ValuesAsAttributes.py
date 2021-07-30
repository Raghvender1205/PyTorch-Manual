import math
import torch


@torch.jit.script
def fn():
    return math.pi
