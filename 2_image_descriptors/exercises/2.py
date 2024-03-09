import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from mod_harris import *


def main() -> int:
    """ Incrementally apply stronger blur (or ROF de-noising) to an image
        and extract Harris corners. What happens? """

    im = np.array(Image.open('../../data/crans_1_small.jpg').convert('L'))

    plt.figure(figsize=(30, 10))
    plt.gray()

    sigma = 0

    for n_plot in range(8):
        plt.subplot(2, 4, n_plot + 1)

        sigma += 2

        harrisim = compute_harris_response(im, sigma)
        filtered_coords = get_harris_points(harrisim, 30)

        plt.imshow(im)
        plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
        plt.title(f'Sigma: {sigma} --- Harris Points: {len(filtered_coords)}')
        plt.axis('off')

    print('The larger the sigma, the more points are found, but less accurate they are.')
    plt.show()
    return 0


if __name__ == "__main__":
    main()
