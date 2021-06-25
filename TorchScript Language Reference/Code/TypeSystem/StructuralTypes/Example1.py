# This Example uses typing.NamedTuple syntax to define a tuple.

import torch
from typing import NamedTuple, Tuple

class MyTuple(NamedTuple):
    first: int
    second: int
    
def inc(x: MyTuple) -> Tuple[int, int]:
    return (x.first + 1, x.second + 1)

t = MyTuple(first=1, second=2)
scripted_inc = torch.jit.script(inc)
print('TorchScript: ', scripted_inc(t))
