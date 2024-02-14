import numpy as np


def get_circle_points(circle_x, circle_y):
    """ 
    Computes the Bresenham's line algorithm to find the circumference
    pixels of a point with a radius of 3px.
    :param circle_x: x position of the central point.
    :param circle_y: y position of the central point.
    :return: clockwise ordered list with the circumference pixels.
    """
    RADIUS = 3

    circle_points = np.zeros((16, 2), dtype=int)

    # set the first point at (x, y + r)
    x = circle_x
    y = circle_y + RADIUS

    # initialise the value of f_p
    f_p = 1 - RADIUS
    index = 0

    # save the point in the four cuadrants
    circle_points[index] = [x, y]
    circle_points[index + 4] = [y, -x]
    circle_points[index + 8] = [-x, -y]
    circle_points[index + 12] = [x, -y]

    # the algorithm knows when to stop when x = y
    while x <= y:
        # x increments by 1 each iteration
        x += 1
        index += 1

        if f_p <= 0:
            # the point is inside the circle or on the perimeter
            f_p = f_p + 2 * y + 1
        else:
            # the point is outside the circle
            y -= 1
            f_p = f_p + 2 * y - 2 * (x - 1) + 1

        circle_points[index] = [x, y]
        circle_points[index + 4] = [y, -x]
        circle_points[index + 8] = [-x, -y]
        circle_points[index + 12] = [x, -y]

    return circle_points


def get_fast_points(im, n_contiguous_pixels, threshold):
    """
    :return: Corners detected from FAST Corner Detector.
    """
    pass


def high_speed_test(fast_points):
    """
    Test for rejecting non-corner FAST points.
    :return: List with filtered FAST points.
    """
    pass
