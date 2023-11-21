# Лабораторная работа №4
# Вариант 4: y = 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x)

from math import sin, cos
from matplotlib import pyplot
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
layers_config = [(LinearFunction(), 6), (SigmoidFunction(), 2), (LinearFunction(), 1)]
network = NeuralNetwork(layers_config)

inputs = make_sample(0.1)[:30]
predict_references = make_sample(0.1)[30:]

errors = network.back_propagation(inputs, min_error)
results = network.predict(inputs+predict_references)

pyplot.plot(errors)
pyplot.show()
for i in range(len(results)):
    print(f'\nСпрогнозированное значение: {results[i]}\nЭталонное значение: {(inputs+predict_references)[i+6]}'
          f'\nСреднеквадратичное отклонение: {0.5 * ((results[i] - (inputs+predict_references)[i+6]) ** 2)}')
pyplot.plot((inputs + predict_references)[6:])
pyplot.plot(results)
pyplot.show()

