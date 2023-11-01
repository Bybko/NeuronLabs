# Лабораторная работа №4
# Вариант 4: y = 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x)

from math import sin
from neurones_architecture import Neuron, NeuralNetwork


# function for calculate values from sinus-function
def make_inputs(step: float) -> list[float]:
    sinuses = []
    x = 0
    for _ in range(45):
        #sinuses.append(0.4 * sin(0.4 * x) + 0.4)
        x += step
    return sinuses


# main
neurons_num = 3
min_error = 0.0000001

network = NeuralNetwork()
for _ in range(neurons_num):
    network.add_neuron(Neuron())

# :30 означает взять с 0 по 29 из списка, возвращаемого make_inputs(), 30: означает с 30 и до конца
inputs = make_inputs(0.1)[:30]
predict_references = make_inputs(0.1)[30:]

network.train(inputs, min_error)
results = network.predict(inputs, 15)

for i in range(len(results)):
    print(f'\nСпрогнозированное значение: {results[i]}\nЭталонное значение: {inputs[i+3]}'
          f'\nСреднеквадратичное отклонение: {0.5 * ((results[i] - inputs[i + 3]) ** 2)}')
