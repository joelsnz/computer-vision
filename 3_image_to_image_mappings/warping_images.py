import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import Delaunay

import warp


def main() -> int:
    # warping_images()
    # image_in_image()
    # triangulated_image_in_image()
    # piecewise_affine_warping()
    turning_torso()

    return 0


def warping_images() -> None:
    im = np.array(Image.open("../data/empire.jpg").convert("L"))
    H = np.array([[1.4, 0.05, -100], [0.05, 1.5, -100], [0, 0, 1]])
    im2 = ndimage.affine_transform(im, H[:2, :2], (H[0, 2], H[1, 2]))

    plt.figure()
    plt.gray()
    plt.imshow(im2)
    plt.show()

    return


def image_in_image() -> None:
    # example of affine warp of im1 onto im2
    im1 = np.array(Image.open("../data/alcatraz1.jpg").convert("L"))
    im2 = np.array(Image.open("../data/book_frontal.JPG").convert("L"))
    # set to points
    tp = np.array([[264, 538, 540, 264], [40, 36, 605, 605], [1, 1, 1, 1]])
    im3 = warp.image_in_image(im1, im2, tp)

    plt.figure()
    plt.gray()
    plt.imshow(im3)
    plt.axis("off")
    plt.show()

    return


def triangulated_image_in_image() -> None:
    im1 = np.array(Image.open("../data/alcatraz1.jpg").convert("L"))
    im2 = np.array(Image.open("../data/book_frontal.JPG").convert("L"))

    tp = np.array([[264, 538, 540, 264], [40, 36, 605, 605], [1, 1, 1, 1]])

    im4 = warp.advanced_image_in_image(im1, im2, tp)

    plt.figure()
    plt.gray()
    plt.imshow(im4)
    plt.axis("off")
    plt.show()

    return


def piecewise_affine_warping() -> None:
    points = np.array(np.random.standard_normal((100, 2)))
    tri = Delaunay(points)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], "*")
    plt.axis("off")
    plt.show()

    return


def turning_torso() -> None:
    # open image to warp
    fromim = np.array(Image.open("../data/sunset_tree.jpg"))
    x, y = np.meshgrid(range(5), range(6))
    x = (fromim.shape[1] / 4) * x.flatten()
    y = (fromim.shape[0] / 5) * y.flatten()

    # triangulate
    tri = warp.triangulate_points(x, y)

    # open image and destination points
    im = np.array(Image.open("../data/turningtorso1.jpg"))
    tp = np.loadtxt("../data/turningtorso1_points.txt")  # destination points

    # convert points to hom. coordinates
    fp = np.vstack((y, x, np.ones((1, len(x)))))
    tp = np.vstack(
        (tp[:, 1], tp[:, 0], np.ones((1, len(tp)))), dtype="int64", casting="unsafe"
    )

    # warp triangles
    im = warp.pw_affine(fromim, im, fp, tp, tri)

    # plot
    plt.figure()
    plt.imshow(im)
    warp.plot_mesh(tp[1], tp[0], tri)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
