# For type hinting
from typing import List, Tuple

# For image preprocessing
import numpy as np

# For accessing the dataset
from cell_counter.import_dataset import load_synthetic_dataset

# For creating and using CNN
import tensorflow as tf


def cnn_preprocess_data(path:str=None, num:int = 2500) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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
    (training_images, training_labels), ( testing_images, testing_labels,) = load_synthetic_dataset(path=path, num=num, resolution=(128, 128))

    scale = 1 / float(255)
    for index, image in enumerate(training_images):
        training_images[index] = image * scale
    for index, image in enumerate(testing_images):
        testing_images[index] = image * scale

    return (training_images, training_labels), ( testing_images, testing_labels,)


def build_cnn():
    """
    Returns a CNN model for use on a preprocessed dataset.

    Returns:
    keras.engine.sequential.Sequential: The generated CNN.

    """

    preprocessed_image_shape = (128, 128, 1)
    cnn_filter = 16
    cnn_pool_size = 2
    kernal = (3, 3)
    dropout_rate = 0.2

    from tensorflow.keras import Sequential, layers

    model = Sequential()

    model.add(
        layers.Conv2D(
            cnn_filter, kernal, activation="relu", input_shape=preprocessed_image_shape
        )
    )
    model.add(layers.MaxPooling2D(pool_size=cnn_pool_size))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2D(2 * cnn_filter, kernal, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=cnn_pool_size))
    model.add(layers.Conv2D(2 * cnn_filter, kernal, activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(1, activation="relu"))

    return model


def compile_cnn(model):
    """
    Compiles the CNN model.

    Parameters:
    model: The CNN to be compiled.

    """
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])


def run_cnn(
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
    Runs the CNN model.

    Parameters:
    model: The CNN model to run.
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
    ) = cnn_preprocess_data(path=path, num=image_number)
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
    model = build_cnn()
    compile_cnn(model)

    training_hist, _ = run_cnn(
        model, epochs=25, image_number=2500, validation_split=0.1
    )

    import matplotlib.pyplot as plt

    plt.plot(training_hist.history["mse"], label="mse")
    plt.plot(training_hist.history["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.show()
