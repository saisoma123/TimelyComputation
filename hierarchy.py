import math
import networkx as nx
import matplotlib.pyplot as plt

import math

class Node:
    def __init__(self, name=None):
        self.name = name
        self.value = None
        self.outputs = []
        self.inputs = []
        self.visited = False

    def forward(self, t=None):
        raise NotImplementedError

    def reset(self):
        self.visited = False
        self.value = None
        for node in self.outputs:
            node.reset()


class TensorProduct(Node):
    """Combines two nodes in parallel."""
    def __init__(self, node1, node2):
        super().__init__()
        self.inputs = [node1, node2]
    
    def forward(self, t=None):
        return (self.inputs[0].forward(t), self.inputs[1].forward(t))

class Source(Node):
    """Initial object, produces a value without any input."""
    def __init__(self, value_func):
        super().__init__()
        self.value_func = value_func

    def forward(self, t=None):
        return self.value_func(t)

class Sink(Node):
    """Terminal object, consumes a value and produces no output."""
    def __init__(self, consume_func=None):
        super().__init__()
        self.consume_func = consume_func or (lambda x, t: None)

    def forward(self, t=None):
        value = self.inputs[0].forward(t)
        self.consume_func(value, t)
        return None


# Category Theory Concepts

class Morphism(Node):
    def __init__(self, source=None, target=None, name=""):
        super().__init__(name=name)
        self.source = source
        self.target = target

    def compose(self, other):
        if self.source == other.target:
            return CompositeMorphism(self, other)
        else:
            raise ValueError("Morphisms cannot be composed due to mismatched source/target")

class CompositeMorphism(Morphism):
    def __init__(self, first, second):
        super().__init__(source=second.source, target=first.target)
        self.first = first
        self.second = second
        self.inputs = [self.second] + self.first.inputs
        self.name = f"({self.second.name} ∘ {self.first.name})"
    
    def forward(self, t):
        value_first = self.first.forward(t)
        value_second = self.second.forward(t)
        return value_second

class IdentityMorphism(Morphism):
    def __init__(self, source):
        super().__init__(source=source, target=source, name=f"id_{source.name}")

    def forward(self, t):
        return self.source.forward(t)

# Time-Varying Value
class TimeVaryingValue(Node):
    def __init__(self, function, name=""):
        super().__init__(name=name)
        self.function = function

    def forward(self, t):
        return self.function(t)

class Functor:
    def __init__(self, node):
        self.node = node

    def map(self, operation_class, other_functor):
        operation_instance = operation_class(self.node, other_functor.node)
        self.node.outputs.append(operation_instance)
        other_functor.node.outputs.append(operation_instance)
        operation_instance.inputs = [self.node, other_functor.node]
        return Functor(operation_instance)


# Math Operations (as morphisms)
class Add(Morphism):
    def forward(self, t):
        if not self.visited:
            self.value = sum(node.forward(t) for node in self.inputs)
            self.visited = True
        return self.value

class Subtract(Morphism):
    def forward(self, t):
        if not self.visited and len(self.inputs) == 2:
            self.value = self.inputs[0].forward(t) - self.inputs[1].forward(t)
            self.visited = True
        return self.value

class Multiply(Morphism):
    def forward(self, t):
        if not self.visited:
            self.value = 1
            for node in self.inputs:
                value = node.forward(t)
                if isinstance(value, (int, float)):
                    self.value *= value
                else:
                    raise ValueError("Invalid type: expected int or float, got {}".format(type(value)))
            self.visited = True
        return self.value

class Divide(Morphism):
    def forward(self, t):
        if not self.visited and len(self.inputs) == 2:
            numerator = self.inputs[0].forward(t)
            denominator = self.inputs[1].forward(t)
            if denominator == 0:
                raise ValueError("Division by zero!")
            self.value = numerator / denominator
            self.visited = True
        return self.value

def visualize_graph(final_node):
    G = nx.DiGraph()
    
    nodes_to_visit = [final_node]
    visited_nodes = set()
    
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if current_node in visited_nodes:
            continue
        
        visited_nodes.add(current_node)
        
        G.add_node(id(current_node), label=str(current_node.__class__.__name__))
        
        for input_node in current_node.inputs:
            G.add_edge(id(input_node), id(current_node))
            nodes_to_visit.append(input_node)
            
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=15)
    plt.show()

# Testing and Visualization
f1 = Functor(TimeVaryingValue(lambda t: math.sin(t), name="sine_wave"))
f2 = Functor(IdentityMorphism(source=f1.node))
f3 = f2.map(Add, f1)
f3.node.name = "Addition"

start_time = 0
end_time = 2 * math.pi  # one full sine wave cycle
time_step = math.pi / 8  # choose your desired time step

current_time = start_time
while current_time <= end_time:
    value_at_t = f3.node.forward(current_time)
    print(f"Value of Addition node at t={current_time:.2f}: {value_at_t:.2f}")
    f3.node.reset()
    current_time += time_step

visualize_graph(f3.node)

sine_source = Source(lambda t: math.sin(t))
cosine_source = Source(lambda t: math.cos(t))

# Combine these sources into a tensor product
tensor_product = TensorProduct(sine_source, cosine_source)

# Define a sink that consumes the tensor product and prints it
sink = Sink(consume_func=lambda value, t: print(f"At t={t:.2f}, Sin: {value[0]:.2f}, Cos: {value[1]:.2f}"))
sink.inputs = [tensor_product]

# Run the graph for a range of t values
for t in [i * 0.1 for i in range(20)]:
    sink.forward(t)
    
visualize_graph(sink)