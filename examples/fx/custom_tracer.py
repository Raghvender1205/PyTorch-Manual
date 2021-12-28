import torch
from torch.fx import symbolic_trace, Tracer, Graph, GraphModule, Node
from typing import Any, Callable, Dict, Optional, Tuple, Union

"""
How to Create and Use Custom Tracers

`Tracer`--the class that implements the symbolic tracing functionality
of `torch.fx.symbolic_trace`--can be subclassed to override various behaviours of the 
tracing process.

In this, we can make our own custom tracers, customizer symbolic tracing using these custom traces.
By simply overriding some methods in the `Tracer` class, we can alter the Graph produced by symbolic tracing.

NOTE: To call `symbolic_trace(m)` is equivalent to `GraphModule(m, Tracer().trace(m))`.
"""

"""
Custom Tracer #1: Trace through All `torch.nn.ReLU` Submodules

During symbolic tracing, some submodules are traced through and their constituent ops are recorded and other
submodules appear as atomic "call_module" Node in the IR. A module in this latter category is called a "leaf module". 
By default, all modules in the PyTorch standard library (`torch.nn`) are leaf modules. We can change this
by creating a custom Tracer and overriding `is_leaf_module`. In this case, we'll keep the default behavior for all `torch.nn` 
Modules except for `ReLU`.
"""
class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)
    
default_traced: GraphModule = symbolic_trace(M1())
"""
Tracing with the default tracer and calling `print_tabular` produces:

    opcode       name    target    args       kwargs
    -----------  ------  --------  ---------  --------
    placeholder  x       x         ()         {}
    call_module  relu_1  relu      (x,)       {}
    output       output  output    (relu_1,)  {}
"""
default_traced.graph.print_tabular()

class LowerReluTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, qualname: str):
        if isinstance(m, torch.nn.ReLU):
            return False
        return super().is_leaf_module(m, qualname)


"""
Tracing with our custom tracer and calling `print_tabular` produces:

    opcode         name    target                             args       kwargs 
    -------------  ------  ---------------------------------  ---------  ------------------
    placeholder    x       x                                  ()         {}
    call_function  relu_1  <function relu at 0x7f66f7170b80>  (x,)       {'inplace': False}
    output         output  output                             (relu_1,)  {}
"""
lower_relu_tracer = LowerReluTracer()
custom_traced_graph: Graph = lower_relu_tracer.trace(M1())
custom_traced_graph.print_tabular()


"""
Custom Tracer #2: Add an Extra Attribute to Each Node

Here, we'll override `create_node` so that we can add a new attribute to 
each Node during its creation.
"""
class M2(torch.nn.Module):
    def forward(self, a, b):
        return a + b

class TaggingTensor(Tracer):
    def create_node(self, kind: str, target: Union[str, Callable], 
                    args: Tuple[Any], kwargs: Dict[str, Any], name: Optional[str] = None, 
                    type_expr: Optional[Any] = None) -> Node:
        n = super().create_node(kind, target, args, kwargs, name)
        n.tag = 'foo'
        return n

custom_traced_graph: Graph = TaggingTensor().trace(M2())

def assert_all_nodes_have_tags(g: Graph) -> bool:
    for n in g.nodes:
        if not hasattr(n, "tag") or not n.tag == "foo":
            return False
    return True

print(assert_all_nodes_have_tags(custom_traced_graph)) # Should return True