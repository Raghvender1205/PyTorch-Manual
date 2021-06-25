# In this example, only the True branch is evaluated, since a is annotated as final and set to True.

import torch

a: torch.jit.final[bool] = True

if a:
    print(torch.empty(2, 3))
else:
    print([])