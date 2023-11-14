from abc import ABC, abstractmethod
import numpy as np


class ActivateFunction(ABC):
    @abstractmethod
    def activate(self, weighted_sum: float) -> float:
        pass


class LinearFunction(ActivateFunction):
    def __init__(self, linear_coefficient: float = 1):
        self.k = linear_coefficient

    def activate(self, weighted_sum: float) -> float:
        return self.k * weighted_sum


class SigmoidFunction(ActivateFunction):
    def activate(self, weighted_sum: float) -> float:
        return 1 / (1 + np.exp(weighted_sum * (-1)))
