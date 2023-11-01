from random import random


class Neuron:
    def __init__(self, outputs_num: int):
        self.weights = [random() for _ in range(outputs_num)]
        self.x = 0

    def take_input(self, x: float) -> None:
        self.x = x

    # activate-here


class NeuronLayer:
    def __init__(self, input_neuron_num: int, output_neuron_num: int = None):
        self.t = random()

        if output_neuron_num is None:
            output_neuron_num = input_neuron_num

        self.input_neurons = [Neuron(output_neuron_num) for _ in range(input_neuron_num)]
        self.outputs = [0 for _ in range(output_neuron_num)]

    def fill_inputs(self, input_image: list[float]) -> None:
        for neuron, x in zip(self.input_neurons, input_image):
            neuron.take_input(x)

    def make_outputs(self) -> list[float]:
        self.outputs = [0 for _ in range(len(self.outputs))]

        for j in range(len(self.outputs)):
            for i in range(len(self.input_neurons)):
                self.outputs[j] += self.input_neurons[i].weights[j] * self.input_neurons[i].x
            self.outputs[j] -= self.t
        return self.outputs


class NeuralNetwork:
    def __init__(self, layers_config: list[int]):
        self.layers = [NeuronLayer(layers_config[i]) for i in range(len(layers_config))]

    def make_result(self, input_image: list[float]) -> list[float]:
        self.layers[0].fill_inputs(input_image)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].fill_inputs(self.layers[i].make_outputs())
        return self.layers[-1].make_outputs()
