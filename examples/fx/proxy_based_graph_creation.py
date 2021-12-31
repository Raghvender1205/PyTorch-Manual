"""
We will create a Graph using Proxy Objects instead of Tracing

We can directly create a Proxy object around a raw Node. This can be used to create a Graph
independently of symbolic tracing.

Here we will see how to use Proxy with a raw Node to append operations to a fresh Graph. We'll
create two parameters (``x`` and ``y``), perform some operations on those parameters, then add
everything we created to the new Graph.

Then we'll wrap that Graph in a GraphModule. Doing that creates a runnable instance of ``nn.Module``
where previously created operations are represented in the Module's ``forward`` function.

In the end, we'll add the following method to an empty ``nn.Module`` class.

.. code-block:: python
	def forward(self, x, y):
		cat_1 = torch.cat([x, y]); x = y = None
		tanh_1 = torch.tanh(cat_1); cat_1 = None
		neg_1 = torch.neg(tanh_1); tanh_1 = None
		return neg_1
"""
import torch
from torch.fx import Graph, Proxy, GraphModule

# Create a Graph independently of symbolic Tracing 
graph = Graph()

# Create raw Nodes
raw1 = graph.placeholder('x')
raw2 = graph.placeholder('y')

# Initialize Proxies using Raw Nodes
y = Proxy(raw1)
z = Proxy(raw2)

# Create other operations using the Proxies `y` and `z`
a = torch.cat([y, z])
b = torch.tanh(a)
c = torch.neg(b)

"""
Create a new Output Node and add it to the Graph. By doing this, Graph will contain all the Nodes we created
as they all are linked to the output node.
"""
graph.output(c.node)

# Wrap the created Graph in a GraphModule to get a final, runnable instance of ``nn.Module``
mod = GraphModule(torch.nn.Module(), graph)