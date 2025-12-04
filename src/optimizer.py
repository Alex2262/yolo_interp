import torch
import torch.optim as optim
from torchvision import transforms as T
import matplotlib.pyplot as plt


from config import *


class Optimizer:
    def __init__(self, interp):
        self.interp = interp

    def run(self, num_iterations, lr=0.01):

        if self.interp.seed is None:
            raise RuntimeError("No seed image set")

        if INIT_RANDOM:
            self.interp.curr_x = torch.randn_like(self.interp.seed).to(self.interp.device).requires_grad_(True)
        else:
            self.interp.curr_x = self.interp.seed.detach().clone().to(self.interp.device).requires_grad_(True)

        optimizer = optim.Adam([self.interp.curr_x], lr=lr)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            x_transformed = self.interp.curr_x
            for transform in TRANSFORMS:
                x_transformed = transform.apply(x_transformed)

            out = self.interp.model.model(x_transformed)
            loss = self.interp.loss.get(x_transformed)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.interp.curr_x.clamp_(0, 1)

            # prints
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss}")

        self.visualize_result(self.interp.curr_x)

    @staticmethod
    def visualize_result(x):
        optim_img = T.ToPILImage()(x.squeeze(0).detach().cpu())
        plt.imshow(optim_img)
        plt.axis('off')
        plt.title("Optimized Image")
        plt.show()
