import cv2
import torch
import torch.optim as optim
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from config import *
from region_maxxing import REGION

class Optimizer:
    def __init__(self, interp):
        self.interp = interp

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

            x_transformed = self.interp.curr_x
            for transform in TRANSFORMS:
                x_transformed = transform.apply(x_transformed)

            out = self.interp.model.model(x_transformed)
            # print(type(out), len(out), out[0], type(out[0]), out[0].shape)

            loss = self.interp.loss.get(x_transformed, out)

            loss.backward()

            if self.interp.curr_x.grad is not None:
                for transform in TRANSFORMS:
                    pass
                    # self.interp.curr_x.grad.data = transform.undo(self.interp.curr_x.grad.data)

            optimizer.step()

            with torch.no_grad():
                self.interp.curr_x.clamp_(0, 1)

            # prints
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss}")

        self.predict_bbs(self.interp.curr_x)

    @staticmethod
    def visualize_result(x):
        optim_img = T.ToPILImage()(x.squeeze(0).detach().cpu())
        plt.imshow(optim_img)
        plt.axis('off')
        plt.title("Optimized Image")
        plt.show()

    def predict_bbs(self, x, conf_threshold=0.25):
        img = T.ToPILImage()(x.squeeze(0).detach().cpu())
        results = self.interp.model(x, conf=conf_threshold, verbose=False)

        print(f"\n{'=' * 60}")
        print(f"YOLO Predictions (confidence >= {conf_threshold}):")
        print(f"{'=' * 60}")

        if len(results[0].boxes) == 0:
            print("No detections found ):(")
        else:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().item()
                class_id = int(box.cls[0].cpu().item())
                class_name = results[0].names[class_id]

                print(f"\nDetection {i + 1}:")
                print(f"  Class: {class_name} (ID: {class_id})")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"  Center: ({(x1 + x2) / 2:.1f}, {(y1 + y2) / 2:.1f})")
                print(f"  Size: {x2 - x1:.1f} x {y2 - y1:.1f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.imshow(img)
        ax1.set_title('Optimized Image', fontsize=14)
        ax1.axis('off')

        ax2.imshow(img)
        ax2.set_title(f'YOLO Predictions (conf >= {conf_threshold})', fontsize=14)
        ax2.axis('off')

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().item()
            class_id = int(box.cls[0].cpu().item())
            class_name = results[0].names[class_id]

            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax2.add_patch(rect)

            # Add label
            label = f'{class_name} {confidence:.2f}'
            ax2.text(
                x1, y1 - 5, label,
                color='red', fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3')
            )

        plt.tight_layout()
        plt.show()
