import cv2
import torch
import torch.optim as optim


from config import *
from region_maxxing import REGION
from visual_helpers import visualize_result, visualize_result_w_bbs


class Optimizer:
    def __init__(self, interp):
        self.interp = interp
        self.params = {
            "LAMBDA_BASE": 1000,
            "LAMBDA_DISTANCE": 0
        }

        self.transforms = list(INITIAL_TRANSFORMS)

    def run(self, num_iterations, lr):

        if self.interp.seed is None:
            raise RuntimeError("No seed image set")

        if INIT_RANDOM:
            self.interp.curr_x = torch.randn_like(self.interp.seed).to(self.interp.device).requires_grad_(True)
        else:
            self.interp.curr_x = self.interp.seed.detach().clone().to(self.interp.device).requires_grad_(True)

            '''
            x1, y1, x2, y2 = REGION
            mask = torch.ones_like(self.interp.seed, dtype=torch.bool)
            mask[:, :, y1:y2 + 1, x1:x2 + 1] = 0
            self.interp.curr_x = torch.where(mask, self.interp.curr_x, torch.randn_like(self.interp.seed)).detach().requires_grad_(True)
            '''

        optimizer = optim.Adam([self.interp.curr_x], lr=lr)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            x_transformed = self.interp.curr_x.to(self.interp.device)
            for transform in self.transforms:
                x_transformed = transform.apply(x_transformed)

            out = self.interp.model.model(x_transformed)
            # print(type(out), len(out), out[0], type(out[0]), out[0].shape)

            loss = self.interp.loss.get(x_transformed, out, self.params)

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                self.interp.curr_x.clamp_(0, 1)

            # prints
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss}")

        visualize_result_w_bbs(self.interp.model, self.interp.curr_x)
        return self.interp.curr_x
