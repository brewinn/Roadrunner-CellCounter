# Helper functions for importing the dataset

# For type hinting
import numpy as np
from typing import Tuple, List

# For finding the dataset
import os

# For turning the images to arrays
from cell_counter.import_tiff import tiff_to_array

# For random image selection
import random


def load_synthetic_dataset(
        is_random:bool = True, seed:int =None, num: int = 10000, split: float = 0.2, path:str = None
) -> Tuple[Tuple[List[np.array], List[int]], Tuple[List[np.array], List[int]]]:
    """
    Returns two tuples, containing the images and labels for the training and
    test sets, respectively.

    Parameters:
    is_random (bool): If selected images should be randomized
    seed (int|None): Seed for the random images.
    number (int): Total number of images to import from the dataset.
    split (float): The proportion of images to use in the testing set.
    path (str|None): Path to images.

    Returns:
    Tuple[Tuple[List[np.array], List[int]], Tuple[List[np.array], List[int]]]:
    A set of images and labels for the training and testing sets, respectively.

    """

    # Ensure that number requested is more than zero, and does not exceed dataset size
    if num > 19200:
        raise Exception("Number of samples requested exceeds dataset size.")
    if num < 1:
        raise Exception("Number of samples requested must be positive.")

    # Set seed, if given
    if seed and is_random:
        random.seed(seed)

    # Find path to images
    if path:
        path_to_images = path
    else:
        counter_dir = os.path.dirname(__file__)
        path_to_images = counter_dir + "/../resources/BBBC005_v1_images/"

    # List of found tif files
    image_filenames = [
        image_file
        for image_file in os.listdir(path_to_images)
        if image_file[-4:] == ".TIF"
    ]

    # Determine the number of testing and training samples
    num_test = int(num * split)
    if num_test < 0:
        num_test = 0
    elif num_test > num:
        num_test = num
    num_training = num - num_test

    # Randomly select images from dataset
    samples = random.sample(image_filenames, num) if is_random else image_filenames[:num]

    # Convert images to arrays, note that we only need the first 'page'
    # images = [tiff_to_array(path_to_images + sample)[0] for sample in samples]
    images = []
    for sample in samples:
        images.append(tiff_to_array(path_to_images + sample)[0])

    # Find the cell count from the filename
    import re

    def extract_label(sample: str) -> int:
        label = re.match(r".*_.*_C(\d+)", sample)
        if label:
            return int(label.group(1))
        else:
            raise Exception(f"Failed to extract label from {sample}")

    labels = [extract_label(image) for image in samples]

    # Split samples and labels into test and training sets
    training_samples = images[:num_training]
    training_labels = labels[:num_training]
    testing_samples = images[num_training:]
    testing_labels = images[num_training:]

    return (training_samples, training_labels), (testing_samples, testing_labels)
