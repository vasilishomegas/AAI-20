import random

rnd_weight = random.randint(0, 1)

class Node:
    def __init__(self):
        self.prev_nodes = []
        self.value = None

    def add_prev_node(self, node):
        self.prev_nodes.append((node, rnd_weight))

    def get_prev_nodes(self):
        return self.prev_nodes

    def set_value(self, value):
        self.value = value

    def get_value(self, function):
        if self.prev_nodes:
            return function(sum(list(map((lambda x: x[0].get_value(function) * x[1]), self.prev_nodes))))
        else:
            return function(self.value)  # not sure if you need to do this over a input/bias


class NeuralNetwork:
    def __init__(self, network, function, learning_rate):

        self.function = function
        self.learning_rate = learning_rate
        self.input_nodes = []
        self.output_nodes = []

        nodes = dict((x, y) for x, y in (list(map(lambda x: (x, Node()), network))))
        for node in network:
            if not node[1]:
                self.output_nodes.append(nodes[node[0]])
            else:
                for connection in node[1]:
                    nodes[connection].add_prev_node(nodes[node[0]])

    def run(self, inputs):
        zip(self.input_nodes, inputs)

    def train(self, inputs, outputs, repeat = 1):
        return





nn = [(1,[2,3]),(2,[4]),(3,[4]),(4,[])]
