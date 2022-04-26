"""
author: alan cabrera
references:
1.https://arxiv.org/pdf/1505.04597.pdf
2.https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
3.https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

    The convolutional neural network, or CNN,
is a kind of neural network model designed 
to working with two-dimensional image data.
    It make use of a convolutional layer that 
gives the network its name. This layer 
performs an operation called a convolution 
which is essentially taking the dot product
using a set of weights, or filters, and an
array derived from an input image.
    
tensorflow is a machine learning library
keras is a neaural network library
"""
import os
import tensorflow as tf  # keras is integrated into tensorflow, as noted from NuSeT
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from tqdm import tqdm_notebook  # for a progress bar

# For importing the dataset (thanks Brendan!)
from cell_counter.import_dataset import load_synthetic_dataset


def unet_preprocess_data(path: str = None):
    """
    Used to normalize data:
    Normalization in image processing is used to 
    change the intensity level of pixels. It is 
    used to get better contrast in images with poor 
    contrast due to glare.
    Function inspired by group leader Brendan Winn and 
    the following implementation at:
    https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb

    Parameters:
    path (str): Path to images.

    Returns:
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The
    dataset, including the preprocessed images.
    """
    ids = next(os.walk(path))[2]  # list of names all images in the given path

    (training_images, training_labels), (
        testing_images, testing_labels,) = load_synthetic_dataset(path=path, num=len(ids), resolution=(128, 128))

    scale = 1/float(255)
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        training_images[n] = training_images[n] * scale
        testing_images[n] = testing_images[n] * scale

    return (training_images, training_labels), (
        testing_images,
        testing_labels,
    )


def get_unet(pretrained_weights=None, input_size=(256, 256, 1)):
    """
    Returns a unet model. 
    From:
    https://github.com/zhixuhao/unet/blob/master/model.py
    But might change to another source.
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    model = get_unet()
