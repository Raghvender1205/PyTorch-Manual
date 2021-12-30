import torch
import torch.fx

"""
In this example we will define a library of 'composite' operations. Composite operations are those
that are defined as callable functions that are composed of several other operations in their implementation.

Composite operations allow you to choose at what level of abstraction you want to interpret/manipulate the code.
We show that we can provide a function to inline these functions as well as use a custom Tracer to automatically
inline such functions.

Composite operations can be useful for exposing higher-level to a backend/transform while still maintaining the 
ability to examine things at a more fine-grained level.
"""

def sigmoid_lowp(x: torch.Tensor):
    x = x.float()
    return x.half()

"""
wrap() indicates that the passed-in function should always be recorded as a `call_function` node rather than 
being traced through. We would also see.

1) Inline the implementation of such a function and 
2) Define a Tracer that automatically traces through such a function.
"""
torch.fx.wrap(sigmoid_lowp)

def add_lowp(a: torch.Tensor, b: torch.Tensor):
    a, b = a.float(), b.float()
    c = a + b
    return c.half()

torch.fx.wrap(add_lowp)

# Let's see what happens when we symbolically trace through some code that uses these functions
class Foo(torch.nn.Module):
    def forward(self, x, y):
        x = sigmoid_lowp(x)
        y = sigmoid_lowp(y)
        return add_lowp(x, y)

traced = torch.fx.symbolic_trace(Foo())
print(traced.code)
"""
Output would be:

def forward(self, x, y):
    sigmoid_lowp = __main___sigmoid_lowp(x);  x = None
    sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None
    add_lowp = __main___add_lowp(sigmoid_lowp, sigmoid_lowp_1);  sigmoid_lowp = sigmoid_lowp_1 = None
    return add_lowp
"""