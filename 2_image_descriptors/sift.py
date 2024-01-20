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


def match(desc1, desc2):
    """ For each descriptor in the first image,
        select its match in the second image.
        input: desc1 (descriptors for the first image),
        desc2 (same for second image). """
    
    # normalize vectors to unit length
    desc1 = np.array([d / np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d / np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i, : ], desc2t) # vector of dot products
        dotprods *= 0.9999
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores


def match_twosided(desc1, desc2):
    """ Two-sided symmetric version of match() """

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12