import random
import math
from math import e

rnd_weight = random.uniform(-1.0, 1)


def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 + (e**(-x)))


def derivative_sigmoid(x):
    # """Expects input x to be already sigmoid-ed""" NOPE
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (e ** (2*x) - 1) / (e ** (2*x) + 1)


def derived_tanh(x):
    # """Expects input x to already be tanh-ed.""" NOPE
    return 1 - tanh(x)*tanh(x)


class Neuron:
    def __init__(self):
        self.prev_neurons = []
        self.weights = []
        self.next_neurons = []
        self.a = None
        self.delta = None
        self.z = None
        self.bias = None
        self.output_goal = None
        
    def calculate_z(self):
        self.z = 0
        for x in range(len(self.prev_neurons)):
            self.z += self.prev_neurons[x].calculate_z * self.weights[x]
        return self.z
    
    def calculate_output_delta(self):
        self.delta = (self.output_goal-self.a)*derivative_sigmoid(self.z)
        return self.delta
    
    def calculate_delta(self):
        self.delta = 0
        for neuron in self.next_neurons:
            neuron.calculate_delta()
        derivative_sigmoid(self.z)

    def add_prev_neuron(self, neuron):
        self.prev_neurons.append((neuron, rnd_weight))

    def get_prev_neurons(self):
        return self.prev_neurons

    def set_value(self, value):
        self.a = value

    def get_value(self, function): # calculate a
        if self.prev_neurons:
            self.a = function(sum(list(map((lambda x: x[0].get_value(function) * x[1]), self.prev_neurons))))
        return self.a


class NeuralNetwork:
    def __init__(self, network, function, learning_rate):

        self.function = function
        self.learning_rate = learning_rate
        self.output_neurons = []

        neurons = dict(map(lambda x: (x[0], Neuron()), network))
        for neuron in network:
            if not neuron[1]:
                self.output_neurons.append(neurons[neuron[0]])
            else:
                for connection in neuron[1]:
                    neurons[connection].add_prev_neuron(neurons[neuron[0]])
        self.input_neurons = [neuron[1] for neuron in neurons.items() if not neuron[1].get_prev_neurons()]

    def run(self, inputs):
        for neuron, value in zip(self.input_neurons, inputs):
            neuron.set_value(value)
        return list(map(lambda n: n.get_value(self.function), self.output_neurons))

    def train(self, inputs, outputs, repeat=1):
        return


def main():
    network_structure = [(1, [2, 3]), (2, [4]), (3, [4]), (4, [])]
    activation_function = math.atanh

    nn = NeuralNetwork(network_structure, activation_function, 0.1)
    print(nn.input_neurons)
    print(nn.output_neurons)
    print(nn.run([1]))


if __name__ == '__main__':
    main()

