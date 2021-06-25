import torch
from typing import List

def g(l: List[int], val: int):
    l.append(val)
    return l

def f(val: int):
    l = g(torch.jit.annotate(List[int], []), val)
    return l


m = torch.jit.script(f)
print("Eager: ", f(3))
print("TorchScript: ", m(3))