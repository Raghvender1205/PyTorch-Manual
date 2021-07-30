import torch

from typing import Tuple, Any

@torch.jit.export
def inc_first_element(x: Tuple[int, Any]):
    return (x[0] + 1, x[1])


m = torch.jit.script(inc_first_element)
print(m((1, 2.0)))
print(m((1, (100, 200))))