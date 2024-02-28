import numpy as np


def get_bresenham_circle(circle_x, circle_y):
    """ 
    Computes the Bresenham's line algorithm to find the circumference
    pixels of a point.
    :param radius_px: radius of the circle in pixels.
    :return: clockwise ordered list with the circumference pixels.
    """
    RADIUS = 3

    actual_x = circle_x + RADIUS
    actual_y = circle_y

    circle_points = np.empty([16, 2])
    circle_points[0][0] = actual_x
    circle_points[0][1] = actual_y

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
