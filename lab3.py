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
    def __init__(self, a, min_e, neurons_num: int):
        self.a = a
        self.minError = min_e
        self.neurons_nums = neurons_num
        self.t = random()
        self.y = 0
        self.S = 0

        self.neurons = []
        for _ in range(neurons_num):
            network.add_neuron(Neuron())

    def add_neuron(self, neuron) -> None:
        self.neurons.append(neuron)
        self.neurons_nums += 1

    def activate(self) -> float:
        k = 1
        return k * self.S

    def calculate_output(self, input_image: list[float]) -> float:
        self.fill_input(input_image)
        self.S = 0
        for neuron in self.neurons:
            self.S = self.S + neuron.x * neuron.weight

        self.S -= self.t
        self.y = self.activate()
        return self.y

    def calculate_error(self, reference: float) -> float:
        return 0.5 * ((self.y - reference) ** 2)

    def fill_input(self, input_image: list[float]) -> None:
        for neuron, x in zip(self.neurons, input_image):
            neuron.take_input(x)

    def train(self, inputs: list[float]) -> None:
        epochs = 0
        optimal = False
        while not optimal:
            error = 0
            print(f'\nЭпоха {epochs + 1}:')
            for i in range(len(inputs) - self.neurons_nums):
                input_image = inputs[i:i+self.neurons_nums] #от i включительно до i+self.neurons_num не включительно
                reference = inputs[i+self.neurons_nums]

                self.calculate_output(input_image)

                print(f'Значение: {self.y}\nЭталонное значение: {reference}')

                error += self.calculate_error(reference)

                for neuron in self.neurons:
                    neuron.weight = neuron.weight - self.a * neuron.x * (self.y - reference)
                self.t = self.t + self.a * (self.y - reference)
            epochs += 1
            if error < self.minError:
                optimal = True

    def predict(self, inputs: list[float], duration: int) -> list[float]:
        results = []
        window = [inputs[0], inputs[1], inputs[2]]
        for _ in range(duration):
            buffer_element = self.calculate_output(window)
            for i in range(len(window) - 1):
                window[i] = window[i + 1]
            window[-1] = buffer_element
            results.append(window[-1])
        return results


# function for calculate values from sinus-function
def make_inputs(step: float) -> list[float]:
    sinuses = []
    x = 0
    for _ in range(45):
        sinuses.append(4 * sin(8 * x) + 0.4)
        x += step
    return sinuses


# main
neurons_num = 3
network = NeuralNetwork(0.001, 0.0000001, neurons_num)

inputs = make_inputs(0.1)[:30]
predict_references = make_inputs(0.1)[30:]

network.train(inputs)
results = network.predict(inputs, 15)

for i in range(len(results)):
    print(f'\nСпрогнозированное значение: {results[i]}\nЭталонное значение: {inputs[i+3]}')
