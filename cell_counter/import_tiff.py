# Simple wrapper for reading in and normalizing a tiff image

# For type hinting
import numpy as np
from typing import List

# For reading in the tiff
import cv2

# Take in the path to the file, and return a List of arrays representing the
#  image.
def tiff_to_array(path:str)->List[np.array]:
    # The following string is a 'docstring',
    # it can be seen with help(tiff_to_array)
    """
    Returns a list of arrays representing the tiff at the given path.

    Parameters:
    path (int): The absolute path to the file. A relative directory may be
    supplied, but may not work as intended.

    Returns:
    List[np.array]: A list of arrays representing the different pages of the
    tiff.

    """

    # Read in the tiff and return its associated arrays
    images = []
    _, images = cv2.imreadmulti(mats = images, filename = path, flags = cv2.IMREAD_GRAYSCALE)
    return images

