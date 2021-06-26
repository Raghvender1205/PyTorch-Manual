# type annotations for Python 3
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class EmptyDataStructures(nn.Module):
    def __init__(self):
        super(EmptyDataStructures, self).__init__()
    
    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[int, float]], Dict[str, int]]:
        # This annotates the list to be a ```List[Tuple[int, float]]```
        my_list: List[Tuple[int, float]] = []
        for i in range(10):
            my_list.append((i, x.item()))
        
        my_dict: Dict[str, int] = {}
        return my_list, my_dict
    
x = torch.jit.script(EmptyDataStructures())