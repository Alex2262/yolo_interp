
import torch
from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def apply(self, x):
        pass


class Jitter(Transform):
    def __init__(self, amount: int):
        self.amount = amount

    def apply(self, x):
        shift_y = torch.randint(-self.amount, self.amount + 1, (1,)).item()
        shift_x = torch.randint(-self.amount, self.amount + 1, (1,)).item()

        x_j = torch.roll(x, shifts=shift_y, dims=2)
        x_j = torch.roll(x_j, shifts=shift_x, dims=3)
        return x_j

