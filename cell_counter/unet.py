"""
author: alan cabrera
references:
1.https://arxiv.org/pdf/1505.04597.pdf (the original research paper)
2.https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
3.https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
4.https://github.com/ashishrana160796/nalu-cell-counting/blob/master/exploring-cell-counting/model.py

    The convolutional neural network, or CNN,
is a kind of neural network model designed 
to work with two-dimensional image data.
    It makes use of a convolutional layer that 
gives the network its name. This layer 
performs an operation called a convolution, 
which is essentially taking the dot product
of a set of weights, or filters, and an
array derived from an input image.
    U-Net, introduced in 2015, was an innovative
approach to addressing the issue of image
segmentation...

We use tensorflow, a machine learning library,
and keras, a neaural network library, to help
make it possible.
"""
# For type hinting
from typing import List, Tuple

# For image preprocessing
import numpy as np

# For accessing the dataset
from cell_counter.import_dataset import get_dataset_info, load_images_from_dataframe

# For creating and using CNN
import tensorflow as tf

# For unet
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet_preprocess_data(path: str = None, num: int = 2500):
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
    df = get_dataset_info(path)
    df = df[df['blur'] == 1]

    # Randomly select 'num' from the remaining images, without replacement
    df = df.sample(n=num, replace=False)

    # Return images and labels from dataframe
    (training_images, training_labels), (
        testing_images, testing_labels,
        ) = load_images_from_dataframe(df, path=path, resolution=(128, 128))

    scale = 1 / float(255)
    for index, image in enumerate(training_images):
        training_images[index] = image * scale
    for index, image in enumerate(testing_images):
        testing_images[index] = image * scale

    return (training_images, training_labels), (
        testing_images,
        testing_labels,
    )

# The original U-Net implementation.
# from https://github.com/zhixuhao/unet/blob/master/model.py

def build_unet():
    """
    Returns a unet model for use on a preprocessed dataset.

    Returns:
    keras.engine.sequential.Sequential: The generated CNN.

    """

    preprocessed_image_shape = (128, 128, 1)

    inputs = Input(pretrained_weights=None,
                   input_size=preprocessed_image_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # =========================================================================
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # =========================================================================
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # =========================================================================
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # =========================================================================
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # =========================================================================
    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)
    # =========================================================================
    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)
    # =========================================================================
    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)
    # =========================================================================
    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    # =========================================================================
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# The modified U-Net architecture, which truncates the decoding 
# part(where upsampling occurs) and uses output from the encoder
# section (where downsampling occurs) as input to a simple CNN
# in order to get a count of the segemented objects.
#
# based on https://github.com/zhixuhao/unet/blob/master/model.py

def build_modified_unet():
    """
    Returns a unet model for use on a preprocessed dataset.

    Returns:
    keras Model class: The generated U-Net codel.

    """
    preprocessed_image_shape = (128, 128, 1)

    inputs = Input(preprocessed_image_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # =========================================================================
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # =========================================================================
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # =========================================================================
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # =========================================================================
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # =========================================================================
    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)
    # =========================================================================
    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)
    # =========================================================================
    # Append simple cnn to reduce the final output to appropriate dimensions as 
    # suggested by my team leader, Brendan Winn. Implementated in 
    # nalu_fcrn_cellcounter.py by Brendan Winn. Modified to fit U-Net.
    
    cnn_filter = 16
    cnn_pool_size = 2
    kernal = (3, 3)
    dropout_rate = 0.2

    outputs_ = Conv2D(cnn_filter, kernal, activation="relu")(conv7)
    outputs_ = MaxPooling2D(pool_size=cnn_pool_size)(outputs_)
    outputs_ = Dropout(dropout_rate)(outputs_)
    outputs_ = Conv2D(
        2 * cnn_filter, kernal, activation="relu")(outputs_)
    outputs_ = MaxPooling2D(pool_size=cnn_pool_size)(outputs_)
    outputs_ = Conv2D(
        2 * cnn_filter, kernal, activation="relu")(outputs_)

    outputs_ = Flatten()(outputs_)
    outputs_ = Dense(64)(outputs_)
    outputs_ = Dense(1, activation="relu")(outputs_)
    
    model = Model(input=inputs, output = outputs_)
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
    model = build_modified_unet()
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
