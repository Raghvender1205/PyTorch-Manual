# Example (refining types on parameters and locals):

import torch
import torch.nn as nn
from typing import Optional


class M(nn.Module):
    z: Optional[int]

    def __init__(self, z):
        super(M, self).__init__()
        # If `z` is None, its type cannot be inferred, so it must be Specified 
        self.z = z
    
    def forward(self, x, y, z):
        # type: (Optional[int], Optional[int], Optional[int]) -> int
        if  x is None:
            x = 1
            x += 1
        
        # Refinement for an attribute by assigning it to a local
        z = self.z
        if y is not None and z is not None:
            x = y + z
        
        # Refinement via an `assert`
        assert z is not None
        x += z
        return x

module = torch.jit.script(M(2))
module = torch.jit.script(M(None))
