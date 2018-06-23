import numpy as np

class Node(object):
	def __init__(self, inbound_nodes = None):
		if inbound_nodes == None:
			self.inbound_nodes = []
		else:
			self.inbound_nodes = inbound_nodes
		self.outbound_nodes = []
		for n in self.inbound_nodes:
			n.outbound_nodes.append(self)
		self.value = None
		self.gradients = {}
	
	def forward(self):
		raise NotImplementedError

	def backward(self):
		raise NotImplementedError

class Input(Node):
	def __init__(self):
		Node.__init__(self)

	def forward(self, value = None):
		if value is not None:
			self.value = value

class Add(Node):
	def __init__(self, x_list):
		Node.__init__(self, x_list)

	def forward(self):
		self.value = 0.0
		for node in self.inbound_nodes:
			self.value += node.value

class Linear(Node):
	def __init__(self, inputs_weights_bias):
		Node.__init__(self, [inputs, weights, bias])

	def forward(self):
		x = np.array(self.inbound_nodes[0].value)
		w = np.array(self.inbound_nodes[1].value)
		b = np.array(self.inbound_nodes[2].value)
		self.value = np.dot(w, x) + b

def forward_pass(output_node, sorted_nodes):
	'''
	perform a forward pass through a list of sorted nodes
	arguments:
		output_node: the output node of the graph (no outgoing edges)
		sorted_nodes: a topologically sorted list of nodes
	returns the output node's value
	'''
	for n in sorted_nodes:
		n.forward()

	return output_node.value

class Sigmoid(Node):
	def __init__(self, x):
		Node.__init__(self, [x])

	def _sigmoid(self, x):
		return 1.0/(1.0+ np.exp(-x))
	
	def forward(self):
		x = self.inbound_nodes[0].value
		self.value = self._sigmoid(x)


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value


def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

x, y, z = Input(), Input(), Input()

f = Add((x, y, z))

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
