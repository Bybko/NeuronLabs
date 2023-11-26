from neurones_architecture import Neuron
from activate_functions import ActivateFunction


class HopfieldNeuron(Neuron):
    def __init__(self, activate_function: ActivateFunction, index: int):
        super().__init__(activate_function)
        self.index = index

    def make_connections(self, neurons_num: int) -> None:
        for _ in range(neurons_num):
            self.weights.append(0)

    def changing_weights(self, input_image: list[int]) -> None:
        for i in range(len(self.weights)):
            self.weights[i] += (2 * self.x - 1) * (2 * input_image[i] - 1)
        self.weights[self.index] = 0

    def hopfield_neuron_output(self, input_image: list[int]) -> float:
        weighted_sum = 0
        for i in range(len(self.weights)):
            weighted_sum += self.weights[i] * input_image[i]
        return self.function.activate(weighted_sum)

    def neuron_output(self) -> float:
        pass

    def neuron_output_hardmax(self, weighted_sums: list[float]) -> float:
        pass

    def calculate_error(self, past_errors: list[float], past_outputs: list[float], function: ActivateFunction) -> None:
        pass


class HopfieldNeuronNetwork:
    def __init__(self, activate_function: ActivateFunction, input_length: int):
        self.neurons = []
        for i in range(input_length):
            self.neurons.append(HopfieldNeuron(activate_function, i))
            self.neurons[i].make_connections(input_length)

    def train(self, input_list: list[int]) -> None:
        for neuron, input_element in zip(self.neurons, input_list):
            neuron.take_input(input_element)
            neuron.changing_weights(input_list)

    def make_result(self, input_image: list[int]) -> list[int]:
        results = []
        for neuron in self.neurons:
            results.append(neuron.hopfield_neuron_output(input_image))

        if results != input_image:
            return self.make_result(results)
        else:
            return results
