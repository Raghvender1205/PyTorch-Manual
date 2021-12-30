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

# Notice that the calls `sigmoid_lowp` and `add_lowp` appear literally in the trace, they are not traced through

# ********** Inlining Calls **************
# Define a Function that allows for inlining these calls during graph maipulation
def inline_lowp_func(n: torch.fx.Node):
    # If we find a call to a function in our `lowp` module, inline it.
    if n.op == 'call_function' and n.target.__module__ == inline_lowp_func.__module__:
        # We want to insert the operations comprising the implementation of the function before the function itself.
        # Then, we can swap the output value of the function call with the output value for its implementation nodes.
        with n.graph.inserting_before(n):
            # We can inline the code by using `fx.Proxy` instances.
            # `map_arg` traverses all aggregate types and applies the given function to Node instances in data structure.
            # In this case, we are applying the `fx.Proxy` constructor.
            proxy_args = torch.fx.node.map_arg(n.args, torch.fx.Proxy)
            proxy_kwargs = torch.fx.node.map_arg(n.kwargs, torch.fx.Proxy)
            # Call the function itself with proxy args. This will emit nodes in the graph corresponding to the
            # operations in the implementation of the function.
            output_proxy = n.target(*proxy_args, **proxy_kwargs)
            # Now replace the original node's uses with the output node of the implementation
            node.replace_all_uses_with(output_proxy.node)
            # Delete the old node
            node.graph.erase_node(node)


for node in traced.graph.nodes:
    if node.op == 'call_function' and node.target is sigmoid_lowp:
        inline_lowp_func(node)

# Recompile after Graph manipulation
traced.recompile()
print(traced.code)
