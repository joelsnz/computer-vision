from PIL import Image
import numpy as np
from numpy import linalg
import homography
from scipy import ndimage

def alpha_for_triangle(points, m, n):
    """ Creates alpha map of size (m, n)
        for a triangle with corners defined by points
        (given in normalized homogeneous coordinates). """

    alpha = np.zeros((m, n))
    for i in range(min(points[0]), max(points[0])):
        for j in range(min(points[1]), max(points[1])):
            x = linalg.solve(points, [i, j, 1])
            if min(x) > 0: # all coefficients positive
                alpha[i, j] = 1

    return alpha


def image_in_image(im1, im2, tp):
    """ Put im1 in im2 with an affine transformation
        such that corners are as close to tp as possible.
        tp are homogeneous and counter-clockwise from top left. """

    # points to warp from
    m, n = im1.shape[:2]
    fp = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # compute affine transformation and apply
    H = homography.Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2, :2],
                                     (H[0, 2], H[1, 2]), im2.shape[:2])
    alpha = im1_t > 0

    return (1 - alpha) * im2 + (alpha * im1_t)


def advanced_image_in_image(im1, im2, tp):
    """ Put im1 in im2 with two affine transformations
        to ensure all corners are on tp.
        tp are homogeneous and counter-clockwise from top left. """

    # set from points to corners of im1
    m, n = im1.shape[:2]
    fp = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # first triangle
    tp2 = tp[:, :3]
    fp2 = fp[:, :3]

    # compute H
    H = homography.Haffine_from_points(tp2, fp2)
    im1_t = ndimage.affine_transform(im1, H[:2, :2],
                                     (H[0, 2], H[1, 2]), im2.shape[:2])

    # alpha for triangle
    alpha = alpha_for_triangle(tp2, im2.shape[0], im2.shape[1])
    im3 = (1 - alpha) * im2 + (alpha * im1_t)

    # second triangle
    tp2 = tp[:, [0, 2, 3]]
    fp2 = fp[:, [0, 2, 3]]

    # compute H
    H = homography.Haffine_from_points(tp2, fp2)
    im1_t = ndimage.affine_transform(im1, H[:2, :2],
                                     (H[0, 2], H[1, 2]), im2.shape[:2])

    # alpha for triangle
    alpha = alpha_for_triangle(tp2, im2.shape[0], im2.shape[1])

    return (1 - alpha) * im3 + (alpha * im1_t)
