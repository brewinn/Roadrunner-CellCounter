# Helper functions for importing the dataset

# For type hinting
import numpy as np
from typing import Tuple, List

# For finding the dataset
import os

# For turning the images to arrays
from cell_counter.import_tiff import tiff_to_array, reduce_resolution

# For random image selection
import random


def load_synthetic_dataset(
    is_random: bool = True,
    seed: int = None,
    num: int = 2000,
    split: float = 0.2,
    path: str = None,
    resolution: Tuple[int,int] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns two tuples, containing the images and labels for the training and
    test sets, respectively.

    Parameters:
    is_random (bool): If selected images should be randomized
    seed (int): Seed for the random images.
    num (int): Total number of images to import from the dataset.
    split (float): The proportion of images to use in the testing set.
    path (str): Path to images.
    resolution (Tuple[int,int]): Resolution to reduce images down to.

    Returns:
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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

    # To keep the images consistent between non-random runs on different
    # machines.
    if not is_random:
        image_filenames.sort()

    # Determine the number of testing and training samples
    num_test = int(num * split)
    if num_test < 0:
        num_test = 0
    elif num_test > num:
        num_test = num
    num_training = num - num_test

    # Randomly select images from dataset
    samples = (
        random.sample(image_filenames, num) if is_random else image_filenames[:num]
    )

    # Convert images to arrays, note that we only need the first 'page'
    # images = [tiff_to_array(path_to_images + sample)[0] for sample in samples]
    images = []
    for sample in samples:
        image = tiff_to_array(path_to_images + sample)[0]
        if resolution != None:
            image = reduce_resolution(image, resolution)
        images.append(image)
        # Should consider implementing some sort of progress bar

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
    training_samples = np.array(images[:num_training])
    training_labels = np.array(labels[:num_training])
    testing_samples = np.array(images[num_training:])
    testing_labels = np.array(labels[num_training:])

    return (training_samples, training_labels), (testing_samples, testing_labels)
