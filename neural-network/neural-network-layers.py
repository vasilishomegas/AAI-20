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
    def __init__(self, function, deriv_function):
        self.a = None
        self.delta = None
        self.z = None
        self.bias = random.uniform(-1.0, 1)
        self.function = function
        self.deriv_function = deriv_function

    # def calculate_z(self, prev_neurons):
    #     self.z = self.bias
    #     for x in range(len(prev_neurons)):
    #         self
    #         self.z += prev_neurons[x].calculate_z * self.weights[x]
    #
    # def calculate_a(self):
    #     if self.prev_neurons:  # if not an input neuron -> probably needs a rework, maybe just don't call this function on an input neuron and check this in network?
    #         self.a = self.function(self.z)
    #
    # def calculate_delta(self):
    #
    #
    # def calculate_bias(self, learning_rate):
    #     if self.prev_neurons:
    #         self.bias += learning_rate*self.delta


class NeuralNetwork:
    def __init__(self, network, function, derivative_function, learning_rate):

        self.learning_rate = learning_rate
        self.network = []
        self.weights = {}

        for layer in network:  # create neurons in the structure of the network
            self.network.append([Neuron(function, derivative_function) for _ in range(layer)])

        for prev_layer, layer in zip(self.network[:-1],self.network[1:]):  # connect neurons in layers
            for neuron in layer:
                for prev_neuron in prev_layer:
                    self.weights[(prev_neuron, neuron)] = random.uniform(-1.0, 1)

    def calculate_weight(self, connection): # THICC
        self.weights[connection] += self.learning_rate * connection[1].delta * connection[0].a

    def calculate_bias(self, neuron: Neuron):
        neuron.bias += self.learning_rate*neuron.delta

    def calculate_z(self, neuron: Neuron, prev_layer):
        neuron.z = neuron.bias
        for prev_neuron in prev_layer:
            neuron.z += self.weights[(prev_neuron, neuron)]*prev_neuron.a

    def calculate_a(self, neuron: Neuron):
        neuron.a = neuron.function(neuron.z)

    def calculate_output_delta(self, neuron: Neuron, y):
        neuron.delta = (y-neuron.a)*neuron.deriv_function(neuron.z)

    def calculate_delta(self, neuron: Neuron, next_neurons):
        deltasum = 0
        for next_neuron in next_neurons:
            deltasum += next_neuron.delta*self.weights[(neuron, next_neuron)]
        neuron.delta = neuron.deriv_function(neuron.z) * deltasum

    def run(self, inputs):
        for input_neuron, value in zip(self.network[0], inputs):
            input_neuron.a = value
        for prev_layer, layer in zip(self.network[:-1], self.network[1:]):
            for neuron in layer:
                self.calculate_z(neuron, prev_layer)
                self.calculate_a(neuron)

    def train(self, inputs_list, outputs_list):
        for inputs, outputs in zip(inputs_list, outputs_list):
            self.run(inputs)
            for neuron, output in zip(self.network[-1], outputs):
                # calculate the error of output neurons here
                self.calculate_output_delta(neuron, output)
            for layer, next_layer in reversed([zip(self.network[1:-1], self.network[2:])]):
                for neuron in layer:  # calculate delta
                    self.calculate_delta(neuron, next_layer)
            for prev_layer, layer in zip(self.network[:-1], self.network[1:]):
                for neuron in layer:  # update weight and biases
                    for prev_neuron in prev_layer:
                        self.calculate_weight((prev_neuron, neuron))
                    self.calculate_bias(neuron)


