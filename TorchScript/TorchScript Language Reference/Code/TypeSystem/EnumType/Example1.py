# This Example defines the class Color as an Enum type:
import torch
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2


def enum_fn(x: Color, y: Color) -> bool:
    if x == Color.RED:
        return True
    return x == y


m = torch.jit.script(enum_fn)

print("Eager: ", enum_fn(Color.RED, Color.GREEN))
print("TorchScript: ", m(Color.RED, Color.GREEN))
