# Лабораторная работа №7
# Вариант 4: вектора 4, 3, 8

from neurones_architecture import NeuralNetwork

vector4 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
vector3 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
vector8 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]

inputs = [vector4, vector3, vector8]
network = NeuralNetwork(len(inputs))

network.train(inputs)
print(network.make_result(vector4))
print(network.make_result(vector3))
print(network.make_result(vector8))
