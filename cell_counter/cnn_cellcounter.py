# For type hinting
from typing import List, Tuple

# For image preprocessing
import numpy as np

# For accessing the dataset
from cell_counter import import_dataset

# For creating and using CNN
import tensorflow as tf


def cnn_preprocess_data(
    dataset: Tuple[Tuple[List[np.array], List[int]], Tuple[List[np.array], List[int]]]
):
    """
    Reduce the resolution, and normalize the images in the dataset.
    Modification is in-place.

    Parameters:
    dataset: The dataset, including the images to be modified.

    """
    training_images = dataset[0][0]
    testing_images = dataset[1][0]

    # Reduce image resolution by averaging, should probably move this to import_tiff
    def reduce_resolution(image: np.array, shape: Tuple[int, int] = (128, 128)):
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

    scale = 1 / float(255)
    for index, image in enumerate(training_images):
        training_images[index] = reduce_resolution(image) * scale
    for index, image in enumerate(testing_images):
        testing_images[index] = reduce_resolution(image) * scale


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
    image_number: int = 10000,
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
    ) = import_dataset.load_synthetic_dataset(path=path, num=image_number)
    dataset = (training_images, training_labels), (testing_images, testing_labels)
    cnn_preprocess_data(dataset)
    if checkpointing:
        if not checkpoint_path:
            import os

            counter_dir = os.path.dirname(__file__)
            checkpoint_path = counter_dir + "/../resources/checkpoints/cnn_cp.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1
        )
        training_history = model.fit(
            np.array(training_images),
            np.array(training_labels),
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[cp_callback],
            verbose=verbose,
        )

    else:
        training_history = model.fit(
            np.array(training_images),
            np.array(training_labels),
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose,
        )

    testing_evaluation = model.evaluate(
        np.array(testing_images),
        np.array(testing_labels),
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

    plt.plot(training_hist.history["mse"], label="mean_squared_error")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.show()
