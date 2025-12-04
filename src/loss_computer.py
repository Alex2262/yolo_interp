

from config import *


class LossComputer:
    def __init__(self, interp):
        self.interp = interp
        self.get = self.get_seeded if USE_SEED else self.get_basic

    def get_basic(self, x):
        if self.interp.targets.get is None:
            raise RuntimeError("No targets configured")

        targets = self.interp.targets.get()
        activation_loss = -targets.mean()

        return activation_loss

    def get_seeded(self, x):
        activation_loss = self.get_basic(x)

        distance = torch.norm(x - self.interp.seed)

        loss = activation_loss + 0.02 * distance

        # print(distance, activation_loss, loss)

        return loss
