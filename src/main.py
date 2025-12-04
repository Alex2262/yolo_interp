

from yolo_interp import YoloInterp


def main():
    y = YoloInterp(device="cpu")

    y.print_layers()

    y.set_seed("../images/dog.jpg")
    y.targets.set_conv1(0)
    y.optimizer.run(100, 0.01)


if __name__ == '__main__':
    main()
