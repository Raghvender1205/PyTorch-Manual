"""
How to replace one Op with Another

1. Iterate through all nodes in your GraphModule's Graph.
2. Determine if the current Node should be replaced. (NOTE: Match on the Node's ``target`` attribute.)
3. Create a replacement Node add it to the Graph.
4. Use the FX built in ``replace_all_uses_with`` to replace all uses of the current Node with the replacement.
5. Delete the old Node from the Graph
6. Call ``recompile`` on the GraphModule. This updates the generated Python code to reflect the new Graph State.


Currently, FX does not provide any way to guarantee that replaced operators are syntatically valid. It's 
up to the user to confirm that any new operators will work with the existing operands.

In this example, we replace any instance of addition with a bitwise AND.

To examine how the Graph evolves during op replacement, add the statement `print(traced.graph)` after the line 
to inspect. Alternatively, call `traced.graph.print_tabular()`to see the IR in a Tablular form.
"""
import torch
from torch.fx import symbolic_trace
import operator

# Sample Module
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)

# Symbolically trace an instance of the module
traced = symbolic_trace(M())

# As demonstrated in the above example, there are several different ways
# to denote addition. The possible cases are:
#     1. `x + y` - A `call_function` Node with target `operator.add`.
#         We can match for equality on that `operator.add` directly.
#     2. `torch.add(x, y)` - A `call_function` Node with target
#         `torch.add`. Similarly, we can match this function directly.
#     3. `x.add(y)` - The Tensor method call, whose target we can match
#         as a string.

patterns = set([operator.add, torch.add, "add"])

# Go through all the nodes in the Graph
for n in traced.graph.nodes:
    # If the target matches one of the patterns
    if any(n.target == pattern for pattern in patterns):
        # Set the insert point, add the new node, and replace all uses.
        # of `n` with a new node.
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        # Remove the old node from the Graph
        traced.graph.erase_node(n)

# Recompile
traced.recompile()
print(traced.graph)
"""
Tabular Form

opcode         name           target                                                              args                                            kwargs
-------------  -------------  ------------------------------------------------------------------  ----------------------------------------------  --------
placeholder    x              x                                                                   ()                                              {}
placeholder    y              y                                                                   ()                                              {}
call_function  bitwise_and    <built-in method bitwise_and of type object at 0x00007FF8B5F8A590>  (x, y)                                          {}
call_function  bitwise_and_1  <built-in method bitwise_and of type object at 0x00007FF8B5F8A590>  (x, y)                                          {}
call_function  bitwise_and_2  <built-in method bitwise_and of type object at 0x00007FF8B5F8A590>  (x, y)                                          {}
output         output         output                                                              ((bitwise_and, bitwise_and_1, bitwise_and_2),)  {}
None

Simple Output `traced.graph`
graph():
    %x : [#users=3] = placeholder[target=x]
    %y : [#users=3] = placeholder[target=y]
    %bitwise_and : [#users=1] = call_function[target=torch.bitwise_and](args = (%x, %y), kwargs = {})
    %bitwise_and_1 : [#users=1] = call_function[target=torch.bitwise_and](args = (%x, %y), kwargs = {})
    %bitwise_and_2 : [#users=1] = call_function[target=torch.bitwise_and](args = (%x, %y), kwargs = {})
    return (bitwise_and, bitwise_and_1, bitwise_and_2)
"""
