import os
from PIL import Image
import numpy as np

def get_imlist(path):
    """ Returns a list of filenames for all jpg images in a directory. """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    """ Resize an image array using PIL. """
    pil_im = Image.fromarray(np.uint8(im))

    return np.array(pil_im.resize(sz))


def histeq(im, nbr_bins=256):
    """ Histogram equalization of a grayscale image """

    # get image histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


def compute_average(imlist):
    """ Compute the average of a list of images. """

    # open first image and make into array of type float
    averageim = np.array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)

    #return average as uint8
    return np.array(averageim, 'uint8')