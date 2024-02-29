from fast import *


def main() -> int:
    """ An alternative corner detector to Harris is the FAST
        corner detector. There are a number of implementations
        including a pure Python version available at
        http://www.edwardrosten.com/work/fast.html.
        Try this detector, play with the sensitivity threshold,
        and compare the corners with the ones from our Harris
        implementation. """

    points = []
    points = get_fast_descriptors(3, 2)
    print(points)

    return 0


if __name__ == "__main__":
    main()
