import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mod_harris import *


def main() -> int:
    """Modify the function for matching Harris corner points to also take
    a maximum pixel distance between points for them to be considered
    as correspondences in order to make matching more robust."""

    im1 = np.array(Image.open("../../data/crans_1_small.jpg").convert("L"))
    im2 = np.array(Image.open("../../data/crans_2_small.jpg").convert("L"))

    wid = 5
    harrisim = compute_harris_response(im1, 5)
    filtered_coords1 = get_harris_points(harrisim, wid + 1)
    d1 = get_descriptors(im1, filtered_coords1, wid)

    harrisim = compute_harris_response(im2, 5)
    filtered_coords2 = get_harris_points(harrisim, wid + 1)
    d2 = get_descriptors(im2, filtered_coords2, wid)

    print("Starting matching")
    matches = match_twosided(d1, d2, distance=30.0)

    plt.figure()
    plt.gray()
    plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
