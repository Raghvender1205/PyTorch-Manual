# ERROR: Tuple not recognized because not imported from typing 

import torch
from typing import Tuple

@torch.jit.export
def inc(x: Tuple[int, int]):
    return (x[0] + 1, x[1] + 1)

m = torch.jit.script(inc)
print(m((1, 2)))
