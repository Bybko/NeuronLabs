# Лабораторная работа №2
# Вариант 4: y = 4 * sin(8 * x) + 0.4

from random import random
from math import sin


class Neuron:
    def __init__(self):
        self.weight = random()
        self.x = 0

    def take_input(self, x: float):
        self.x = x


class NeuralNetwork:
    def __init__(self, a, min_e, neurons: list[Neuron]):
        self.a = a
        self.minError = min_e
        self.t = random()
        self.y = 0
        self.S = 0
        self.neurons = neurons

    def add_neuron(self, neuron) -> None:
        self.neurons.append(neuron)

    def activate(self) -> float:
        k = 1
        return k * self.S

    def calculate_output(self, input: list[float]) -> float:
        self.fill_input(input)
        self.S = 0
        for neuron in self.neurons:
            self.S = self.S + neuron.x * neuron.weight

        self.S -= self.t
        self.y = self.activate()
        return self.y

    def calculate_error(self, inputs: list[float], reference: float) -> float:
        error = 0
        for _ in range(len(inputs)):
            error += (self.y - reference) ** 2
        return 0.5 * error

    def fill_input(self, inputs: list[float]) -> None:
        for neuron, x in zip(self.neurons, inputs):
            neuron.take_input(x)

    def train(self, inputs: list[float], reference: float) -> None:
        optimal = False
        while not optimal:
            self.calculate_output(inputs)
            error = self.calculate_error(reference)

            if error < self.minError:
                optimal = True

            for neuron in self.neurons:
                neuron.weight = neuron.weight - self.a * neuron.x * (self.y - reference)
            self.t = self.t + self.a * (self.y - reference)

    def predict(self, inputs: list[float], duration: int) -> None:
        window = inputs
        for _ in range(duration):
            buffer_element = self.calculate_output(window)
            for i in range(len(window) - 1):
                window[i] = window[i + 1]
            window[-1] = buffer_element


def make_inputs(step: float) -> list[float]:
    sinuses = []
    x = 0
    for _ in range(45):
        sinuses.append(4 * sin(8 * x) + 0.4)
        x += step
    return sinuses


# main
neurones = [Neuron(), Neuron(), Neuron()]
network = NeuralNetwork(0.1, 0.00001, neurones)

inputs = make_inputs(0.1)[:30]
inputs = [inputs[i:i+3] for i in range(27)]
references = make_inputs(0.1)

for input, reference in zip(inputs, references):
    network.train(input, reference)

network.predict(inputs[-1], 15)


