# For Loop on Tuples

import torch
from typing import Tuple

@torch.jit.script
def fn():
    tup = (3, torch.ones(4))
    for x in tup:
        print(x)

if __name__ == '__main__':
    fn()