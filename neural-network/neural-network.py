import random
import math

rnd_weight = random.uniform(-1.0, 1)


class Neuron:
    def __init__(self):
        self.prev_neurons = []
        self.value = None

    def add_prev_neuron(self, neuron):
        self.prev_neurons.append((neuron, rnd_weight))

    def get_prev_neurons(self):
        return self.prev_neurons

    def set_value(self, value):
        self.value = value

    def get_value(self, function):
        if self.prev_neurons:
            self.value = function(sum(list(map((lambda x: x[0].get_value(function) * x[1]), self.prev_neurons))))
        return self.value


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

