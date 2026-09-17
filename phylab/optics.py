"""Interactive tools for processing diffraction-pattern photographs."""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["photoproc", "photocalib"]


def photoproc(flnm):
    """Extract a linear intensity profile from a diffraction-pattern photo.

    Displays the image and prompts the user to click the two opposite
    corners of a crop box around the pattern of interest.

    Parameters
    ----------
    flnm : str
        Path to the JPEG (or other image) file containing the pattern.

    Returns
    -------
    numpy.ndarray
        The normalised, calibrated intensity profile of the cropped
        pattern.
    """
    img = mpimg.imread(flnm)
    fig = plt.figure()
    plt.imshow(img)

    print("---------------------------------------------")
    print("Note: Click two corners to crop image.")
    print("---------------------------------------------")

    pts = plt.ginput(2)
    pts = np.array(pts).astype(int)
    plt.close(fig)

    # Order corners from top-left to bottom-right.
    box_pts = np.zeros((2, 2), dtype=int)
    if pts[0, 0] > pts[1, 0]:
        box_pts[:, 0] = [pts[1, 0], pts[0, 0]]
    else:
        box_pts[:, 0] = pts[:, 0]

    if pts[0, 1] > pts[1, 1]:
        box_pts[:, 1] = [pts[1, 1], pts[0, 1]]
    else:
        box_pts[:, 1] = pts[:, 1]

    cropped = img[box_pts[0, 1]:box_pts[1, 1], box_pts[0, 0]:box_pts[1, 0], :]
    plt.imshow(cropped)  # display for checking the cropping job

    intensity = np.sum(np.sum(cropped, axis=0), axis=1).astype(float)

    # Photographic response calibration.
    intensity = intensity ** 1.64

    normalised = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    return normalised


def photocalib(flnm, width):
    """Determine the physical size of a pixel from a photo of a ruler.

    Displays the image and prompts the user to click both ends of a
    horizontal ruler or measurement mark of known length.

    Parameters
    ----------
    flnm : str
        Path to the JPEG (or other image) file showing the ruler.
    width : float
        The known physical length of the marked distance, in whatever
        units the caller wants the result expressed in.

    Returns
    -------
    float
        The physical width represented by one pixel, in the same units
        as ``width``.
    """
    img = mpimg.imread(flnm)
    fig = plt.figure()
    plt.imshow(img)

    print("---------------------------------------------")
    print("Note: Click both ends of rule distance. Make sure measurement is horizontal.")
    print("---------------------------------------------")

    pts = plt.ginput(2)
    pts = np.array(pts).astype(int)
    plt.close(fig)

    pixel_distance = np.abs(pts[0, 0] - pts[1, 0])
    return width / pixel_distance
