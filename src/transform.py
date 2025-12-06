
import torch
from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def apply(self, x):
        pass

    @abstractmethod
    def undo(self, x):
        pass


class Jitter(Transform):
    def __init__(self, amount: int):
        self.amount = amount
        self.shift_y = None
        self.shift_x = None

    def apply(self, x):
        self.shift_y = torch.randint(-self.amount, self.amount + 1, (1,)).item()
        self.shift_x = torch.randint(-self.amount, self.amount + 1, (1,)).item()

        x_j = torch.roll(x, shifts=(self.shift_y, self.shift_x), dims=(2, 3))
        return x_j

    def undo(self, x):
        return torch.roll(x, shifts=(-self.shift_y, -self.shift_x), dims=(2, 3))
