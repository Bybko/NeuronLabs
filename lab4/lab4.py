# Лабораторная работа №4
# Вариант 4: y = 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x)

from math import sin, cos
from neurones_architecture import NeuralNetwork
from activate_functions import LinearFunction, SigmoidFunction


def make_sample(step: float) -> list[float]:
    values = []
    x = 0
    for _ in range(45):
        values.append(0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x))
        x += step
    return values


min_error = 0.0001
a = 0.0001
layers_config = [(LinearFunction(), 6), (SigmoidFunction(), 2), (LinearFunction(), 1)]
network = NeuralNetwork(layers_config, a)

inputs = make_sample(0.1)[:30]
predict_references = make_sample(0.1)[30:]

print(network.make_result([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

network.back_propagation(inputs, min_error)


