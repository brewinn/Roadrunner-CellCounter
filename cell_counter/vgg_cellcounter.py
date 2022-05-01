"""
Title: VGG16 model
From: https://github.com/uestcsongtaoli/vgg_net/blob/master/train.py
Modified: By William Wells
For: University of Texas at San Antonio
     Artificial Intelligence 
"""

# For type hinting
from typing import List, Tuple

# For image preprocessing
import numpy as np
import pandas as pd

# For accessing the dataset
from cell_counter.import_dataset import get_dataset_info, load_images_from_dataframe

# For creating and using CNN
import tensorflow as tf

from keras.layers import Input, Conv2D, MaxPool2D, ZeroPadding2D, Flatten, Dense, Dropout, BatchNormalization, \
    Activation
from keras.models import Model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def vgg_preprocess_data(
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
    ) = load_images_from_dataframe(df, path=path, resolution=(224, 224), split=split)

    scale = 1 / float(255)
    for index, image in enumerate(training_images):
        training_images[index] = image * scale
    for index, image in enumerate(testing_images):
        testing_images[index] = image * scale

    return (training_images, training_labels), (
        testing_images,
        testing_labels,
    )


def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def build_vgg(input_shape=(224, 224, 1), num_classes=1, weight_path=None):
    """
    Description: Creates a VGG model 
    Returns a VGG model for use on a preprocessed dataset.
    """
    # instantiate a Keras tensor
    input_layer = Input(shape=input_shape)
    # stage 1
    x = conv_block(input_layer, filters=64, kernel_size=(3, 3), name="conv1_1_64_3x3_1")
    x = conv_block(x, filters=64, kernel_size=(3, 3), name="conv1_2_64_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_1_2x2_2")(x)
    # stage 2
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_1_128_3x3_1")
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_2_128_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_2_2x2_2")(x)
    # stage 3
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_1_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_2_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(1, 1), name="conv3_3_256_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_3_2x2_2")(x)
    # stage 4
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv4_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_4_2x2_2")(x)
    # stage 5
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv5_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_5_2x2_2")(x)

    # FC layers
    # FC layer 1
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) #changed from 0.5
    x = Activation("relu")(x)
    # FC layer 2
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) #changed from 0.5
    x = Activation("relu")(x)
    # FC layer 3
    x = Dense(num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x) #changed softmax to relu

    if weight_path:
        x.load_weights(weight_path)
    model = Model(input_layer, x, name="VGG16_Net")
    return model


def compile_vgg(model):
    """
    Compiles the VGG model.
    Parameters:
    model: The VGG to be compiled.
    """
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])


def run_vgg(
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
    Runs the VGG model.
    Parameters:
    model: The VGG model to run.
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
    ) = vgg_preprocess_data(path=path, num=image_number, df=df)
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
    (_,_),(test_im, test_lab) = vgg_preprocess_data(df=df, split=1)
    results = model.evaluate(test_im, test_lab, batch_size=1)

    # Test against random images
    df = get_dataset_info()
    df = df.sample(n=50, replace=False)

    print("Evaluating against random images...")
    (_,_),(test_im, test_lab) = vgg_preprocess_data(df=df, split=1)
    results = model.evaluate(test_im, test_lab, batch_size=1)


if __name__ == "__main__":
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Low training size, no blur')
    model = build_vgg()
    compile_vgg(model)
    training_hist, _ = run_vgg(
        model, epochs=10, image_number=250, validation_split=0.1, checkpointing=False
    )
    evaluate_model(model, 'vgg_l_n')


    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('High training size, no blur')
    model = build_vgg()
    compile_vgg(model)
    training_hist, _ = run_vgg(
        model, epochs=10, image_number=1000, validation_split=0.1, checkpointing=False
    )
    evaluate_model(model, 'vgg_h_n')

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Low training size, random blur')
    model = build_vgg()
    compile_vgg(model)
    df = get_dataset_info()
    df = df.sample(n=250, replace=False)
    training_hist, _ = run_vgg(
        model, epochs=10, image_number=250, validation_split=0.1, checkpointing=False, df=df
    )
    evaluate_model(model, 'vgg_l_r')

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('High training size, random blur')
    model = build_vgg()
    compile_vgg(model)
    df = get_dataset_info()
    df = df.sample(n=1000, replace=False)
    training_hist, _ = run_vgg(
        model, epochs=10, image_number=1000, validation_split=0.1, checkpointing=False, df=df
    )
    evaluate_model(model, 'vgg_h_r')
