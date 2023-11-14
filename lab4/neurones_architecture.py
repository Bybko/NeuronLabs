from random import random
from activate_functions import ActivateFunction


class Neuron:
    def __init__(self, activate_function: ActivateFunction):
        self.weights = []
        self.x = 0
        self.function = activate_function

    def make_connections(self, outputs_num: int):
        for _ in range(outputs_num):
            self.weights.append(random())

    def take_input(self, x: float) -> None:
        self.x = x

    def neuron_output(self) -> float:
        return self.function.activate(self.x)


class NeuronLayer:
    def __init__(self, activate_function: ActivateFunction, input_neuron_num: int):
        self.t = random()
        self.function = activate_function
        self.outputs = []
        self.neurons = [Neuron(self.function) for _ in range(input_neuron_num)]

    def count_outputs(self, output_neuron_num: int):
        for _ in range(output_neuron_num):
            self.outputs.append(0)
        for neuron in self.neurons:
            neuron.make_connections(output_neuron_num)

    def fill_inputs(self, input_image: list[float]) -> None:
        for neuron, x in zip(self.neurons, input_image):
            neuron.take_input(x)

    def make_outputs(self) -> list[float]:
        self.outputs = [0 for _ in range(len(self.outputs))]
        for j in range(len(self.outputs)):
            for i in range(len(self.neurons)):
                self.outputs[j] += self.neurons[i].weights[j] * self.neurons[i].x
            self.outputs[j] -= self.t
        return self.outputs


class NeuralNetwork:
    def __init__(self, layers_config: list[tuple[ActivateFunction, int]]):
        self.layers = []

        for config in layers_config:
            activation_function_type, neuron_num = config
            self.layers.append(NeuronLayer(activation_function_type, neuron_num))
        self.connect_layers()

    def connect_layers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].count_outputs(len(self.layers[i + 1].neurons))
        self.layers[-1].count_outputs(len(self.layers[-1].neurons))

    def make_result(self, input_image: list[float]) -> list[float]:
        self.layers[0].fill_inputs(input_image)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].fill_inputs(self.layers[i].make_outputs())
        return self.layers[-1].make_outputs()
