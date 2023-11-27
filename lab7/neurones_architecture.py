from random import random


class Neuron:
    def __init__(self):
        self.weights = []
        self.x = 0
        self.error = 0

    def make_connections(self, inputs_num: int) -> None:
        for _ in range(inputs_num):
            self.weights.append(random())


class NeuralNetwork:
    def __init__(self, references_num: int):
        self.t = 1
        self.neurons = []
        for i in range(references_num):
            self.neurons.append(Neuron())

    def make_result(self, input_image: list[int]) -> int:
        d = []
        for _ in range(len(self.neurons)):
            d.append(0)
        for i in range(len(self.neurons)):
            for j in range(len(input_image)):
                d[i] += (input_image[j] - self.neurons[i].weights[j]) ** 2
            d[i] = d[i] ** 0.5

        neuron_winner = d.index(min(d))
        return neuron_winner

    def train(self, inputs: list[list[int]]) -> None:
        for neuron in self.neurons:
            neuron.make_connections(len(inputs[0]))

        epochs = 0
        while epochs < 20:
            for input_image in inputs:
                d = []
                for _ in range(len(self.neurons)):
                    d.append(0)
                for i in range(len(self.neurons)):
                    for j in range(len(input_image)):
                        d[i] += (input_image[j] - self.neurons[i].weights[j]) ** 2
                    d[i] = d[i] ** 0.5

                neuron_winner = d.index(min(d))
                self.changing_weights(neuron_winner, input_image)
            epochs += 1

    def changing_weights(self, winner_index: int, input_image: list[int]) -> None:
        train_speed = 1 / self.t
        for i in range(len(input_image)):
            self.neurons[winner_index].weights[i] = self.neurons[winner_index].weights[i] + train_speed * \
                                                    (input_image[i] - self.neurons[winner_index].weights[i])
        self.t += 1
