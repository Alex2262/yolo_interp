

import torch

from loss_computer import LossComputer
from transform import Jitter
from yolo_interp import YoloInterp

from region_maxxing import verify


class Auto:

    def __init__(self, k):
        self.interp = YoloInterp(device="cpu")
        self.k = k

    def run_phase1(self, layers):
        layer_channel_indices = []
        for layer in layers:
            layer_channel_indices.append((layer, None))

        self.interp.targets.set_layers(layer_channel_indices)

        with torch.no_grad():
            self.interp.model.model(self.interp.seed)
            orig_activations = self.interp.targets.get()

        self.interp.optimizer.run(1000, 0.005)

        with torch.no_grad():
            self.interp.model.model(self.interp.curr_x)
            post_activations = self.interp.targets.get()

        new_layers = {}

        for i in range(len(post_activations)):
            layer_idx = layers[i]
            layer_change = post_activations[i] - orig_activations[i]
            change_channels = torch.sum(layer_change, dim=(1, 2))

            change_list = []

            # s = 0
            # for channel in range(change_channels.shape[0]):
            #     s += abs(change_channels[channel].item())

            for channel in range(change_channels.shape[0]):
                score = change_channels[channel].item()  # / s
                change_list.append((score, layer_idx, channel))

            change_list = sorted(change_list, key=lambda x: x[0], reverse=True)

            for j in range(min(self.k, change_channels.shape[0])):
                change = change_list[j][0]
                channel = change_list[j][2]

                print(change, layer_idx, channel)
                if layer_idx not in new_layers:
                    new_layers[layer_idx] = []

                new_layers[layer_idx].append(channel)

        new_layer_channel_indices = []
        for layer, channel_indices in new_layers.items():
            new_layer_channel_indices.append((layer, channel_indices))

        return new_layer_channel_indices

    def run_phase2(self, layer_channel_indices):
        self.interp.loss.get = self.interp.loss.get_basic
        self.interp.targets.set_layers(layer_channel_indices)
        self.interp.optimizer.run(1000, 0.005)

    def auto_layer(self, layers):
        self.interp.set_seed("images/road.jpg")
        layer_channel_indices = self.run_phase1(layers)

        print("Found", layer_channel_indices)

        self.run_phase2(layer_channel_indices)

    def bin(self, layer, channel):
        def test():
            self.interp.set_seed("images/road.jpg")
            self.interp.targets.set_conv_layer(layer, channel)

            with torch.no_grad():
                self.interp.model.model(self.interp.seed)
                orig_activations = self.interp.targets.get()

            x = self.interp.optimizer.run(300, 0.01)

            with torch.no_grad():
                out = self.interp.model.model(x)
                post_activations = self.interp.targets.get()

            c = torch.sum(post_activations - orig_activations).item()

            p = verify(x, out).item()

            if p >= 0.1:
                return True, p, c
            else:
                return False, p, c

        lo = 0
        hi = 1000
        eps = 1

        self.interp.optimizer.params["LAMBDA_BASE"] = hi
        orig_change = test()[2]

        best_scale = 0
        best_change = 0

        while lo + eps < hi:
            mid = (lo + hi) / 2

            print("RANGE:", lo, mid, hi)

            self.interp.optimizer.params["LAMBDA_BASE"] = mid

            flag, prob, change = test()

            print(flag, prob, change)

            if flag:
                lo = mid
                best_scale = mid
                best_change = change
            else:
                hi = mid

        prop_change = best_change / orig_change

        print(best_scale, best_change, orig_change, prop_change)
        return prop_change

    def search_layer(self, layer):
        num_channels = self.interp.layers[layer].conv.weight.shape[0]

        probs = []

        for channel in num_channels:
            p = self.bin(layer, channel)
            probs.append([p, channel])

        sorted(probs, key=lambda x: x[0], reverse=True)






