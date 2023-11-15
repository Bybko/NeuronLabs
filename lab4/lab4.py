# Лабораторная работа №4
# Вариант 4: y = 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x)

from math import sin, cos
from matplotlib import pyplot
from neurones_architecture import NeuralNetwork
from activate_functions import LinearFunction, SigmoidFunction


def make_sample(step: float) -> list[float]:
    values = []
    x = 0
    for _ in range(150):
        values.append(0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x))
        x += step
    return values


min_error = 0.001
layers_config = [(LinearFunction(), 6), (LinearFunction(), 2), (LinearFunction(), 1)]
network = NeuralNetwork(layers_config)

inputs = make_sample(0.1)[:100]
predict_references = make_sample(0.1)[100:]

network.back_propagation(inputs, min_error)
results = network.predict(inputs+predict_references, 150)
pyplot.plot(inputs + predict_references)
pyplot.plot(results)
pyplot.show()
for i in range(len(results)):
    print(f'\nСпрогнозированное значение: {results[i]}\nЭталонное значение: {inputs[i+6]}')

