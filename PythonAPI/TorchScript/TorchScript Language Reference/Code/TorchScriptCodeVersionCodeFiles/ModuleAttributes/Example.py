from typing import List, Dict
import torch
import torch.nn as nn

class Foo(nn.Module):
    # `words` is initialized as an empty list, so its type must be specified.
    words: List[str]

    # The type could potentially be inferred if `a_dict` was not empty,
    # but the annotation ensures `some_dict` will be made into the proper type
    some_dict: Dict[str, int]

    def __init__(self, a_dict):
        super(Foo, self).__init__()
        self.words = []
        self.some_dict = a_dict

        # `int` can be inferred
        self.my_int = 10
    
    def forward(self, x):
        # type: (str) -> int
        self.words.append(x)
        return self.some_dict[x] + self.my_int


f = torch.jit.script(Foo({'hi': 2}))
print(f)