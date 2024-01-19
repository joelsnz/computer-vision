from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5") -> None:
    """ Process an image and save the results in a file. """
    
    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str("sift " + imagename + " --output=" + resultname + " " + params)
    os.system(cmmd)
    print(f"Processed {imagename} to {resultname}")

    return;


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """

    f = np.loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors


def write_features_to_file(filename, locs, desc) -> None:
    """ Save feature location and descriptor to file. """
    np.savetxt(filename, np.hstack((locs, desc)))

    return;


def draw_circle(c, r) -> None:
    t = np.arange(0, 1.01, .01) * 2 * np.pi
    x = r * np.cos(t) + c[0]
    y = r * np.sin(t) + c[1]
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
        plt.plot(locs[:, 0], locs[:, 1], 'ob')
    plt.axis('off')

    return;