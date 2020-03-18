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
        return list(map(lambda neuron: neuron.a, self.network[-1]))

    def train(self, inputs_list, outputs_list):
        integer_list = list(range(len(inputs_list)))
        random.shuffle(integer_list)
        shuffled_list = list(map(lambda i: (inputs_list[i], outputs_list[i]), integer_list))
        for inputs, outputs in shuffled_list:
            self.run(inputs)
            for neuron, output in zip(self.network[-1], outputs):
                # calculate the error of output neurons here
                self.calculate_output_delta(neuron, output)
            for layer, next_layer in reversed(list(zip(self.network[1:-1], self.network[2:]))):
                for neuron in layer:  # calculate delta
                    self.calculate_delta(neuron, next_layer)
            for prev_layer, layer in zip(self.network[:-1], self.network[1:]):
                for neuron in layer:  # update weight and biases
                    for prev_neuron in prev_layer:
                        self.calculate_weight((prev_neuron, neuron))
                    self.calculate_bias(neuron)

def main():

    def convert_classification(i):
        x = [0, 0, 0]
        x[i] = 1
        return x

    temp = list(map(convert_classification, neural_network_classification))  # turn list of classifications into output array
    nn = NeuralNetwork([4, 5, 3], sigmoid, derivative_sigmoid, 0.1)
    for i in range(120):
        nn.train(neural_network_data, temp)  # data imported from file, expected classifications, nr of runs
    num_correct = 0
    print("Show for every value")
    for i in range(len(neural_network_data)):
        result = nn.run(neural_network_data[i])
        print(result, " ", temp[i])
        num_correct += 1 if result.index(max(result)) == temp[i].index(max(temp[i])) else 0
    print()
    print("Succes rate: ", num_correct / len(neural_network_data) * 100, "%")


if __name__ == '__main__':
    main()
