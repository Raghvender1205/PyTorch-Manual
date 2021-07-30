import torch
from torch import Tensor

def foo(a: Tensor, b: int) -> Tensor:
    return a + b

def bar(a):
    fut: torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=2)
    return torch.jit.wait(fut)

script_bar = torch.jit.script(bar)
input = torch.tensor(2)

# only the scripted version executes asynchronously
assert script_bar(input) == bar(input)

# trace is not run asynchronously, but fork is captured in IR
graph = torch.jit.trace(bar, (input,)).graph
assert "fork" in str(graph)

print("Script Bar: ", script_bar)
print(graph)