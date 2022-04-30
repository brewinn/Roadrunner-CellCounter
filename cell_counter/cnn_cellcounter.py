# For type hinting
from typing import List, Tuple

# For image preprocessing
import numpy as np
import pandas as pd

# For accessing the dataset
from cell_counter.import_dataset import get_dataset_info, load_images_from_dataframe

# For creating and using CNN
import tensorflow as tf


def cnn_preprocess_data(
        path: str = None, num: int = 2500, df = pd.DataFrame(), split=0.1
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Reduce the resolution, and normalize the images in the dataset.
    Modification is in-place.

    Parameters:
    path (str): Path to images.
    num (int): Total number of images to import from the dataset.
    df (pd.DataFrame): Images to use, if any.
    split (float): Proportion of images to use for testing.

    Returns:
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The
    dataset, including the preprocessed images.

    """

    if len(df.columns) == 0:
        # Filter to only use images without blur
        df = get_dataset_info(path)
        df = df[df["blur"] == 1]

        # Randomly select 'num' from the remaining images, without replacement
        df = df.sample(n=num, replace=False)

    # Return images and labels from dataframe
    (training_images, training_labels), (
        testing_images,
        testing_labels,
    ) = load_images_from_dataframe(df, path=path, resolution=(128, 128), split=split)

    scale = 1 / float(255)
    for index, image in enumerate(training_images):
        training_images[index] = image * scale
    for index, image in enumerate(testing_images):
        testing_images[index] = image * scale

    return (training_images, training_labels), (
        testing_images,
        testing_labels,
    )


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
    df:pd.DataFrame = pd.DataFrame()
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
    df (pd.DataFrame): Images to use, if any.

    Returns:
    training_history: The model's training statistics.
    testing_evaluation: The model's testing statistics.

    """
    (training_images, training_labels), (
        testing_images,
        testing_labels,
    ) = cnn_preprocess_data(path=path, num=image_number, df=df)
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

def evaluate_model(model, name):
    import matplotlib.pyplot as plt

    plt.plot(training_hist.history["mse"], label="mse")
    plt.plot(training_hist.history["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.savefig(name+'.png')
    plt.clf()

    # Test against blurless images
    df = get_dataset_info()
    df = df[df['blur']==1]
    df = df.sample(n=50, replace=False)

    print("Evaluating against blurless images...")
    (_,_),(test_im, test_lab) = cnn_preprocess_data(df=df, split=1)
    results = model.evaluate(test_im, test_lab, batch_size=1)

    # Test against random images
    df = get_dataset_info()
    df = df.sample(n=50, replace=False)

    print("Evaluating against random images...")
    (_,_),(test_im, test_lab) = cnn_preprocess_data(df=df, split=1)
    results = model.evaluate(test_im, test_lab, batch_size=1)


if __name__ == "__main__":
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Low training size, no blur')
    model = build_cnn()
    compile_cnn(model)
    training_hist, _ = run_cnn(
        model, epochs=10, image_number=250, validation_split=0.1, checkpointing=False
    )
    evaluate_model(model, 'cnn_l_n')


    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('High training size, no blur')
    model = build_cnn()
    compile_cnn(model)
    training_hist, _ = run_cnn(
        model, epochs=10, image_number=1000, validation_split=0.1, checkpointing=False
    )
    evaluate_model(model, 'cnn_h_n')

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Low training size, random blur')
    model = build_cnn()
    compile_cnn(model)
    df = get_dataset_info()
    df = df.sample(n=250, replace=False)
    training_hist, _ = run_cnn(
        model, epochs=10, image_number=250, validation_split=0.1, checkpointing=False, df=df
    )
    evaluate_model(model, 'cnn_l_r')

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('High training size, random blur')
    model = build_cnn()
    compile_cnn(model)
    df = get_dataset_info()
    df = df.sample(n=1000, replace=False)
    training_hist, _ = run_cnn(
        model, epochs=10, image_number=1000, validation_split=0.1, checkpointing=False, df=df
    )
    evaluate_model(model, 'cnn_h_r')
