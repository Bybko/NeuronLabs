from random import random
from activate_functions import ActivateFunction


class Neuron:
    def __init__(self, activate_function: ActivateFunction):
        self.weights = []
        self.x = 0
        self.function = activate_function
        self.error = 0

    def make_connections(self, outputs_num: int) -> None:
        for _ in range(outputs_num):
            self.weights.append(random())

    def take_input(self, x: float) -> None:
        self.x = x

    def neuron_output(self) -> float:
        return self.function.activate(self.x)

    def calculate_error(self, past_errors: list[float], past_outputs: list[float], function: ActivateFunction) -> None:
        self.error = 0
        for i in range(len(self.weights)):
            self.error += past_errors[i] * function.derivative(past_outputs[i]) * self.weights[i]


class Bias(Neuron):
    def __init__(self, activate_function: ActivateFunction):
        super().__init__(activate_function)

    def take_input(self, x: float) -> None:
        pass

    def neuron_output(self) -> float:
        return 1


class NeuronLayer:
    def __init__(self, activate_function: ActivateFunction, input_neuron_num: int):
        self.function = activate_function
        self.outputs = []
        self.t = Bias(self.function)
        self.neurons = [Neuron(self.function) for _ in range(input_neuron_num)]

    def count_outputs(self, output_neuron_num: int) -> None:
        for _ in range(output_neuron_num):
            self.outputs.append(0)
        for neuron in self.neurons:
            neuron.make_connections(output_neuron_num)
        self.t.make_connections(output_neuron_num)

    def fill_inputs(self, input_image: list[float]) -> None:
        for neuron, x in zip(self.neurons, input_image):
            neuron.take_input(x)

    def make_outputs(self) -> list[float]:
        self.outputs = [0 for _ in range(len(self.outputs))]
        for j in range(len(self.outputs)):
            for i in range(len(self.neurons)):
                self.outputs[j] += self.neurons[i].weights[j] * self.neurons[i].neuron_output()
            self.outputs[j] -= self.t.weights[j]
        return self.outputs

    def get_neurons_outputs(self) -> list[float]:
        neuron_outputs = []
        for neuron in self.neurons:
            neuron_outputs.append(neuron.neuron_output())
        return neuron_outputs

    def calculate_errors_on_layer(self, past_errors: list[float], past_outputs: list[float],
                                  layer_function: ActivateFunction) -> None:
        for neuron in self.neurons:
            neuron.calculate_error(past_errors, past_outputs, layer_function)
        self.t.calculate_error(past_errors, past_outputs, layer_function)

    def collect_layer_errors(self) -> list[float]:
        errors_on_the_layer = []
        for neuron in self.neurons:
            errors_on_the_layer.append(neuron.error)
        return errors_on_the_layer


class NeuralNetwork:
    def __init__(self, layers_config: list[tuple[ActivateFunction, int]]):
        self.a = 0
        self.layers = []

        for config in layers_config:
            activation_function_type, neuron_num = config
            self.layers.append(NeuronLayer(activation_function_type, neuron_num))
        self.connect_layers()

    def connect_layers(self) -> None:
        for i in range(len(self.layers) - 1):
            self.layers[i].count_outputs(len(self.layers[i + 1].neurons))
        self.layers[-1].count_outputs(len(self.layers[-1].neurons))

    def make_result(self, input_image: list[float]) -> list[float]:
        self.layers[0].fill_inputs(input_image)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].fill_inputs(self.layers[i].make_outputs())
        return self.layers[-1].make_outputs()

    def calculate_square_error(self, outputs: list[float], references: list[float]) -> float:
        error = 0
        for i in range(len(outputs)):
            error += 0.5 * ((outputs[i] - references[i]) ** 2)
        return error

    def calculate_train_step(self) -> float:
        inputs_sum = 0
        for neuron in self.layers[0].neurons:
            inputs_sum += (neuron.x ** 2)
        return 1 / (1 + inputs_sum)

    def back_propagation(self, inputs: list[list[float]], references: list[list[float]],
                         min_error: float) -> list[float]:
        list_of_errors = []
        epochs = 0
        optimal = False
        while not optimal:
            square_error = 0
            print(f'\nЭпоха {epochs + 1}:')
            for input_image, reference in zip(inputs, references):
                outputs = self.make_result(input_image)
                square_error += self.calculate_square_error(outputs, reference)
                for i in range(len(outputs)):
                    self.layers[-1].neurons[i].error = outputs[i] - reference[i]

                # проходим по всем слоям в обратном направлении, за исключением выходного
                start_index = len(self.layers) - 2
                for i in range(start_index, -1, -1):
                    past_errors = self.layers[i+1].collect_layer_errors()
                    past_outputs = self.layers[i+1].get_neurons_outputs()

                    self.layers[i].calculate_errors_on_layer(past_errors, past_outputs,
                                                             self.layers[i+1].neurons[0].function)

                    self.a = self.calculate_train_step()
                    self.delta_rule(i, past_outputs)

            print(square_error)
            epochs += 1

            list_of_errors.append(square_error)
            if square_error < min_error:
                optimal = True

        return list_of_errors

    def delta_rule(self, layer_index: int, past_outputs: list[float]):
        for i in range(len(self.layers[layer_index].t.weights)):
            self.layers[layer_index].t.weights[i] = self.layers[layer_index].t.weights[i] + \
                                                    self.a * self.layers[layer_index + 1].neurons[i].error * \
                                                    self.layers[layer_index + 1].t.function.derivative(past_outputs[i])

            for neuron in self.layers[layer_index].neurons:
                neuron.weights[i] = neuron.weights[i] - self.a * self.layers[layer_index + 1].neurons[i].error * \
                                    self.layers[layer_index + 1].neurons[i].function.derivative(past_outputs[i]) * \
                                    neuron.neuron_output()
