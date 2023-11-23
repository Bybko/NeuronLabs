# Лабораторная работа №5
# Вариант 4: вектора 4, 3, 8

from matplotlib import pyplot
from neurones_architecture import NeuralNetwork
from activate_functions import LinearFunction, SigmoidFunction, ThresholdFunction

vector4 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
vector3 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
vector8 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]

min_error = 0.001
layers_config = [(LinearFunction(), 20), (SigmoidFunction(), 2), (ThresholdFunction(), 3)]
network = NeuralNetwork(layers_config)

inputs = [vector4, vector3, vector8]
references = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

network.back_propagation(inputs, references, min_error)
print(network.make_result([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]))

