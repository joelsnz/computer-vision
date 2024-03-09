import numpy as np


def get_fast_descriptors(initial_x, initial_y):
    """ 
    Computes the Bresenham's line algorithm to find the circumference
    pixels of a point with a radius of 3px.
    :param circle_x: x_shift position of the central point.
    :param circle_y: y_shift position of the central point.
    :return: clockwise ordered list with the circumference pixels.
    """
    RADIUS = 3

    circle_points = np.zeros((16, 2), dtype=int)

    # set the first point at (x_shift, y_shift + r)
    x_shift = 0
    y_shift = RADIUS

    # initialise the value of f_p
    f_p = 1 - RADIUS

    point_x = initial_x + x_shift
    point_y = initial_y + y_shift
    # save the point in the four cuadrants
    circle_points[x_shift] = [point_x, point_y]
    circle_points[x_shift + 4] = [point_y, -point_x]
    circle_points[x_shift + 8] = [-point_x, -point_y]
    circle_points[x_shift + 12] = [point_x, -point_y]

    # the algorithm knows when to stop when x_shift = y_shift
    while x_shift <= y_shift:
        # x_shift increments by 1 each iteration
        x_shift += 1

        if f_p <= 0:
            # the point is inside the circle or on the perimeter
            f_p = f_p + 2 * y_shift + 1
        else:
            # the point is outside the circle
            y_shift -= 1
            f_p = f_p + 2 * y_shift - 2 * (x_shift - 1) + 1

        point_x = initial_x + x_shift
        point_y = initial_y + y_shift

        circle_points[x_shift] = [point_x, point_y]
        circle_points[x_shift + 4] = [point_y, -point_x]
        circle_points[x_shift + 8] = [-point_x, -point_y]
        circle_points[x_shift + 12] = [point_x, -point_y]

    return circle_points


def get_fast_points(im, threshold):
    """ Return FAST Corner Detector points. """
    candidates = []
    for x_shift in range(im.shape[0]):
        for y_shift in range(im.shape[1]):
            # check if the point is actually a corner

    return candidates


def high_speed_test(fast_points):
    """
    Test for rejecting non-corner FAST points.
    :return: List with filtered FAST points.
    """
    pass
