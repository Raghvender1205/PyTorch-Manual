import torch
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2

@torch.jit.script
def enum_fn(x: Color, y: Color) -> bool:
    if x == Color.RED:
        return True
    return x == y

