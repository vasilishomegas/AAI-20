import random
from math import e
from import_data import neural_network_classification, neural_network_data


def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 + (e**(-x)))


def derivative_sigmoid(x):
    # """Expects input x to be already sigmoid-ed""" NOPE
    x = sigmoid(x)
    return x * (1 - x)


def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (e ** (2*x) - 1) / (e ** (2*x) + 1)


def derived_tanh(x):
    # """Expects input x to already be tanh-ed.""" NOPE

    x = tanh(x)
    return 1 - x*x

class Neuron:
    def __init__(self):
        self.a = None
        self.delta = None

class NeuralNetwork:
    def __init__(self, network, function, derivative_function, learning_rate):

        self.learning_rate = learning_rate
        self.network = []
        self.connections = {}
        self.biases = {}


        for layer in network:
            self.network.append([Neuron() for _ in range(layer)])



        for previous_layer, layer in zip(self.network[:-1],self.network[1:]):
            for neuron in layer:
                for previous_neuron in previous_layer:
                    self.connections[(previous_neuron, neuron)] = random.uniform(-1.0,1)

    def run(self, inputs):
        for input_neuron, value in zip(self.network[0], inputs):
            input_neuron.a = value
        for previous_layer, layer in zip(self.network[:-1], self.network[1:]):
            for neuron in layer:
                for previous_neuron in previous_layer:
                    #  = self.connections[(previous_neuron, neuron)]
                    # calculate a of all nodes here!
                    continue

    def train(self, inputs_list, outputs_list):
        for inputs, outputs in zip(inputs_list, outputs_list):
            self.run(inputs)
            for neuron, output in zip(self.network[-1], outputs):
                # calculate the error of output neurons here
                continue
            for layer in reversed(self.network[:-1]):
                for neuron in layer:
                    # calculate weight and baises
            for layer in self.network:
                for neuron in layer:
                    # update weight and baises



