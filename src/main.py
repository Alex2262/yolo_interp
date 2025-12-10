

import torch
from yolo_interp import YoloInterp
from auto import Auto, binsearch


def main():
    y = YoloInterp(device="cpu")
    y.print_layers()

    y.optimizer.set_initial()
    y.optimizer.run(200, 0.1)

    # y.set_seed("images/road.jpg")
    # y.targets.set_conv_layer(5, [14, 10, 44, 28, 24, 16, 27, 9])
    # y.targets.set_conv_layer(5, [9, 10, 14, 16, 27])
    # y.targets.set_conv_layer(5, [16, 27]) # 16 and 27 are important
    # y.targets.set_conv_layer(20, [2])
    # y.optimizer.run(200, 0.1)

    # y.set_seed("images/road.jpg")
    # y.targets.set_conv_layer(5, [118,  62,  35,  27,  98, 114,  51,  95,  71,  22])
    # y.optimizer.run(1000, 0.005)

    # a = Auto(0)
    # a.sl(20)
    # a.auto_layer([17])
    # a.auto_layer([3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15])
    # binsearch((0, 15))
    # binsearch((5, 9))
    # a.search_layer(5)
    # binsearch((20, 2))
    # binsearch((0, 2))

    # a.search_layer(0)
    # a.search_layer(20)

    # layer 5 channel 9, 10, 14

    # a.bin(5, 2)
    # a.bin_layer(5)


if __name__ == '__main__':
    main()
