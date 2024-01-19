from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


def write_features_to_file(filename, locs, desc) -> None:
    """ Save feature location and descriptor to file. """
    np.savetxt(filename, np.hstack((locs, desc)))

    return;


def draw_circle(c, r) -> None:
    t = np.arange(0, 1.01, .01) * 2 * np.pi
    x = r * np.sin(t) + c[1]
    y = r * np.cos(t) + c[0]
    plt.plot(x, y, 'b', linewidth=2)

    return;


def plot_features(im, locs, circle=False) -> None:
    """ Show image with features. input: im (image as array),
        locs (row, col, scale, orientation of each feature). """

    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:, 1], locs[:, 0], 'ob')
    plt.axis('off')

    return;