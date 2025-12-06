

from yolo_interp import YoloInterp


def main():
    y = YoloInterp(device="cpu")

    y.print_layers()

    # y.set_seed("images/road.jpg")
    # y.targets.set_conv_layer(8, None)
    # y.optimizer.run(1000, 0.01)

    y.set_seed("images/road.jpg")
    y.targets.set_conv_layer(8, None)
    y.optimizer.run(1000, 0.005)


if __name__ == '__main__':
    main()
