import random
from enum import Enum
from math import e
from import_data import neural_network_classification, neural_network_data
rnd_weight = random.uniform(-1.0, 1)


class NeuronState(Enum):
    INIT = 0
    DELTA = 1
    WEIGHT = 2

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
    def __init__(self, function, derivative_function):
        self.prev_neurons = {}
        # self.weights = []
        self.next_neurons = []
        self.a = None
        self.delta = None
        self.z = None
        self.bias = rnd_weight
        self.output_goal = None
        self.function = function
        self.derivative_function = derivative_function
        self.state = None

    def get_next_neurons(self):
        return self.next_neurons

    def set_output_goal(self, target):
        self.output_goal = target

    def calculate_z(self):
        # self.z = 0
        # for x in range(len(self.prev_neurons)):
        #     self.z += self.prev_neurons[x].calculate_z * self.weights[x]
        self.z = sum(map((lambda x: x.get_value() * self.prev_neurons[x]), self.prev_neurons.keys()))
        return self.z
    
    def calculate_delta(self):
        # firstly checks if delta has already been calculated. if so returns delta immediately else
        # checks if it is an output neuron
        # then calculate the delta of the neuron using the corresponding formula
        if self.state != NeuronState.DELTA:
            self.delta = sum(map(Neuron.calculate_delta, self.next_neurons)) * self.derivative_function(self.z) if self.next_neurons else (self.output_goal-self.a)*self.derivative_function(self.z)
            self.state = NeuronState.DELTA
        return self.delta

    def get_weight(self, other):
        return self.prev_neurons[other]

    def set_weight(self, weight, other):
        self.prev_neurons[other] = weight

    def calculate_weight(self, learning_rate):
        # calculate the weight towards each of the next neurons
        # We need to retrieve the weight from the next neuron, as it's stored with the list of previous neurons
        # Afterwards, we also need to write it back to the same next neuron
        for neuron in self.next_neurons:
            neuron.set_weight(neuron.get_weight(self) + learning_rate*self.calculate_delta()*self.a, self)

    def calculate_bias(self, learning_rate):
        self.bias += learning_rate*self.delta
        return

    def add_prev_neuron(self, neuron):
        self.prev_neurons[neuron] = rnd_weight

    def add_next_neuron(self, neuron):
        self.next_neurons.append(neuron)

    def get_prev_neurons(self):
        return self.prev_neurons

    def set_value(self, value):
        self.a = value

    def get_value(self):  # calculate a
        if self.prev_neurons and self.state != NeuronState.INIT:  # if not an input neuron
            self.a = self.function(self.calculate_z() + self.bias)
            self.state = 0
        return self.a

    def check_state(self, state):
        return state != self.state

    def set_state(self, state):
        self.state = state


class NeuralNetwork:
    def __init__(self, network, function, derivative_function, learning_rate, fully_connected=True):

        self.learning_rate = learning_rate
        self.output_neurons = []
        self.network = []
        if fully_connected:
            for layer in network:
                initlayer = []
                for x in range(layer):
                    initlayer.append(Neuron(function, derivative_function))
                self.network.append(initlayer)
            for layer in range(len(self.network)):
                for neuron in self.network[layer]:
                    if layer != 0:
                        # do prev_neurons
                        for prev_neuron in self.network[layer-1]:
                            neuron.add_prev_neuron(prev_neuron)
                    if layer != len(self.network)-1:
                        # do next_neurons
                        for next_neuron in self.network[layer+1]:
                            neuron.add_next_neuron(next_neuron)

        else:
            neurons = dict(map(lambda n: (n[0], Neuron(function, derivative_function)), network))
            for neuron in network:
                if not neuron[1]:
                    self.output_neurons.append(neurons[neuron[0]])
                else:
                    for connection in neuron[1]:
                        neurons[connection].add_prev_neuron(neurons[neuron[0]])  # backward connections
                        neurons[neuron[0]].add_next_neuron(neurons[connection])  # forward connections

            self.input_neurons = [neuron[1] for neuron in neurons.items() if not neuron[1].get_prev_neurons()]

    def run(self, inputs):
        for neuron, value in zip(self.input_neurons, inputs):
            neuron.set_value(value)
        return list(map(lambda n: (n, n.get_value()), self.output_neurons))

    def train(self, inputs, outputs, repeat=1):
        for _ in range(repeat):
            for batch_input, batch_output in zip(inputs, outputs):
                for neuron, target in zip(self.output_neurons, batch_output):
                    neuron.set_output_goal(target)
                self.run(batch_input)
                neuron_queue = self.input_neurons[:]
                for neuron in neuron_queue:
                    if neuron.check_state(NeuronState.WEIGHT):
                        neuron.calculate_weight(self.learning_rate)
                        neuron.calculate_bias(self.learning_rate)
                        neuron_queue += neuron.get_next_neurons()
                        neuron.set_state(NeuronState.WEIGHT)

def main():
    network_structure = [(1, [2, 3]), (2, [4]), (3, [4]), (4, [])]

    print(neural_network_data)
    print(neural_network_classification)

    nn = NeuralNetwork(network_structure, sigmoid, derivative_sigmoid, 0.1, False)
    print(nn.input_neurons)
    print(nn.output_neurons)
    print(nn.train([[1],[]]))


if __name__ == '__main__':
    main()

