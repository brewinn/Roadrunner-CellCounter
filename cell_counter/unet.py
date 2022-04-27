"""
author: alan cabrera
references:
1.https://arxiv.org/pdf/1505.04597.pdf
2.https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
3.https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
4.https://github.com/ashishrana160796/nalu-cell-counting/blob/master/exploring-cell-counting/model.py

    The convolutional neural network, or CNN,
is a kind of neural network model designed 
to work with two-dimensional image data.
    It makes use of a convolutional layer that 
gives the network its name. This layer 
performs an operation called a convolution 
which is essentially taking the dot product
of a set of weights, or filters, and an
array derived from an input image.
    Concatenation is what makes segmentation
possible.
tensorflow is a machine learning library
keras is a neaural network library
"""
# For type hinting
from typing import List, Tuple

# For image preprocessing
import numpy as np

# For accessing the dataset
from cell_counter.import_dataset import get_dataset_info, load_images_from_dataframe

# For creating and using CNN
import tensorflow as tf


def unet_preprocess_data(path: str = None):
    """
    Reduce the resolution, and normalize the images in the dataset.
    Modification is in-place.

    Parameters:
    path (str): Path to images.
    num (int): Total number of images to import from the dataset.

    Returns:
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The
    dataset, including the preprocessed images.

    """
    # Filter to only use images without blur
    df = get_dataset_info()
    df = df[df['blur']==1]

    # Randomly select 'num' from the remaining images, without replacement
    df = df.sample(n=num, replace=False)

    # Return images and labels from dataframe
    (training_images, training_labels), ( testing_images, testing_labels,) = load_images_from_dataframe(df, path=path, resolution=(128, 128))

    scale = 1 / float(255)
    for index, image in enumerate(training_images):
        training_images[index] = image * scale
    for index, image in enumerate(testing_images):
        testing_images[index] = image * scale

    return (training_images, training_labels), (
        testing_images,
        testing_labels,
    )

def build_unet():
    """
    Returns a unet model for use on a preprocessed dataset.

    Returns:
    keras.engine.sequential.Sequential: The generated CNN.

    """

    preprocessed_image_shape = (128, 128, 1)
    

    from tensorflow.keras import Sequential, layers

    

    return model

def compile_unet(model):
    """
    Compiles the unet model.

    Parameters: the model.
    model: The unet to be compiled.

    """
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])


def run_unet(
    model,
    path: str = None,
    image_number: int = 2000,
    checkpointing: bool = True,
    checkpoint_path: str = None,
    epochs: int = 10,
    validation_split: float = 0.0,
    verbose: int = 2,
):
    """
    Runs the unet model.

    Parameters:
    model: The unet model to run.
    path (str): Path to dataset. Defaults to the path described in Usage if none given.
    image_number (int): The number of images to use from the dataset.
    checkpointing (bool): Whether or not to save the model weights.
    checkpoint_path (str): The path to save the checkpoints to.
    epochs (int): Number of epochs to run.
    validation_split (float): Proportion of images to use as a validation set.
    vebose (int): Verbosity of model training and evaluation.

    Returns:
    training_history: The model's training statistics.
    testing_evaluation: The model's testing statistics.

    """
    (training_images, training_labels), (
        testing_images,
        testing_labels,
    ) = unet_preprocess_data(path=path, num=image_number)
    if checkpointing:
        if not checkpoint_path:
            import os

            counter_dir = os.path.dirname(__file__)
            checkpoint_path = counter_dir + "/../resources/checkpoints/cnn_cp.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1
        )
        training_history = model.fit(
            training_images,
            training_labels,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[cp_callback],
            verbose=verbose,
        )

    else:
        training_history = model.fit(
            training_images,
            training_labels,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose,
        )

    testing_evaluation = model.evaluate(
        testing_images,
        testing_labels,
        verbose=0 if verbose == 0 else 1,
    )
    return training_history, testing_evaluation


if __name__ == "__main__":
    model = build_unet()
    compile_unet(model)

    training_hist, _ = run_unet(
        model, epochs=10, image_number=250, validation_split=0.1, checkpointing=False
    )

    import matplotlib.pyplot as plt

    plt.plot(training_hist.history["mse"], label="mse")
    plt.plot(training_hist.history["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.show()