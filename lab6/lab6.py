# Лабораторная работа №6
# Вариант 4: вектора 4, 3, 8

from matplotlib import pyplot
from hopfield_architect import HopfieldNeuronNetwork
from activate_functions import ThresholdFunction

vector4 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
vector3 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
vector8 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]

inputs = [vector4, vector3, vector8]
network = HopfieldNeuronNetwork(ThresholdFunction(), len(inputs[0]))

for input_image in inputs:
    network.train(input_image)
print(network.make_result([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]))

