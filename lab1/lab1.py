# Вариант 4 - Лабораторная работа №1

from random import random


class Neuron:
    def __init__(self):
        self.weight = random()
        self.x = 0

    def take_input(self, x):
        self.x = x


class NeuralNetwork:
    def __init__(self, a, neurons: list[Neuron]):
        self.a = a
        self.t = random()
        self.y = 0
        self.S = 0
        self.neurons = neurons

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def activate(self) -> int:
        if self.S >= 0:
            return 1
        else:
            return -1

    def calculate_output(self, inputs: list[int]) -> int:
        self.fill_input(inputs)
        self.S = 0
        for neuron in self.neurons:
            self.S = self.S + neuron.x * neuron.weight

        self.S -= self.t
        self.y = self.activate()
        return self.y

    def fill_input(self, inputs: list[int]):
        for neuron, x in zip(self.neurons, inputs):
            neuron.take_input(x)

    def train(self, inputs: list[int], reference: int):
        self.calculate_output(inputs)

        if self.y - reference == 0:
            return

        for neuron in self.neurons:
            neuron.weight = neuron.weight - self.a * neuron.x * (self.y - reference)
        self.t = self.t + self.a * (self.y - reference)


# main
neurones = [Neuron(), Neuron()]
references = [-1, -1, -1, 1]
inputs = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
outputs = []
network = NeuralNetwork(0.1, neurones)

while outputs != references:
    outputs = []
    for i in range(len(inputs)):
        network.train(inputs[i], references[i])
    for i in range(len(inputs)):
        outputs.append(network.calculate_output(inputs[i]))

print("Result: ")
print(f"T: {network.t}")
print(f"Values: {outputs}")
print("Weights:")
for neuron in neurones:
    print(neuron.weight)

print(f"For (1,1): {network.calculate_output([1,1])}")
print(f"For (-1,1): {network.calculate_output([-1,1])}")
print(f"For (-1,-1): {network.calculate_output([-1,-1])}")
print(f"For (1,-1): {network.calculate_output([1,-1])}")
