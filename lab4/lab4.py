# Лабораторная работа №4
# Вариант 4: y = 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x)

from math import sin
from neurones_architecture import NeuralNetwork
from activate_functions import LinearFunction, SigmoidFunction, ActivateFunction


# function for calculate values from sinus-function
def make_inputs(step: float) -> list[float]:
    sinuses = []
    x = 0
    for _ in range(45):
        #sinuses.append(0.4 * sin(0.4 * x) + 0.4)
        x += step
    return sinuses


# main
min_error = 0.0000001
layers_config = [(LinearFunction(), 6), (SigmoidFunction(), 2), (LinearFunction(), 1)]
network = NeuralNetwork(layers_config)

# test
print(network.make_result([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))


