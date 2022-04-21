# Simple wrapper for reading in and normalizing a tiff image

# For type hinting
import numpy as np
from typing import List, Tuple

# For reading in the tiff
import cv2

# Take in the path to the file, and return a List of arrays representing the
#  image.
def tiff_to_array(path: str) -> List[np.array]:
    # The following string is a 'docstring',
    # it can be seen with help(tiff_to_array)
    """
    Returns a list of arrays representing the tiff at the given path.

    Parameters:
    path (str): The absolute path to the file. A relative directory may be
    supplied, but may not work as intended.

    Returns:
    List[np.array]: A list of arrays representing the different pages of the
    tiff.

    """

    # Read in the tiff and return its associated arrays
    images = []
    _, images = cv2.imreadmulti(mats=images, filename=path, flags=cv2.IMREAD_GRAYSCALE)
    return images


# Reduce image resolution by averaging
def reduce_resolution(image: np.array, shape: Tuple[int, int] = (128, 128)) -> np.array:
    intermediate_image = np.zeros((shape[0], image.shape[1]))
    new_image = np.transpose(np.zeros(shape))
    ratio = image.shape[0] / float(shape[0])
    value = ratio
    carryover = 0.0
    pixel = 0
    for new_pixel in range(shape[0]):
        while value > 1:
            if carryover > 0:
                intermediate_image[new_pixel] += carryover * image[pixel - 1]
            else:
                intermediate_image[new_pixel] += image[pixel]
                value -= 1
                pixel += 1
        intermediate_image[new_pixel] += value * image[pixel]
        intermediate_image[new_pixel] /= ratio
        carryover = 1 - ratio
        value = ratio

    ratio = image.shape[1] / float(shape[1])
    value = ratio
    carryover = 0.0
    pixel = 0
    intermediate_image = np.transpose(intermediate_image)
    for new_pixel in range(shape[1]):
        while value > 1:
            if carryover > 0:
                new_image[new_pixel] += carryover * intermediate_image[pixel - 1]
                carryover = 0
            else:
                new_image[new_pixel] += intermediate_image[pixel]
                value -= 1
                pixel += 1
        new_image[new_pixel] += value * intermediate_image[pixel]
        new_image[new_pixel] /= ratio
        carryover = 1 - ratio
        value = ratio

    return np.transpose(new_image)
