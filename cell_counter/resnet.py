'''
Amber Beserra Lib322
Referenced from: https://github.com/suvoooo/Learn-TensorFlow/blob/master/resnet/Implement_Resnet_TensorFlow.ipynb
'''
# For type hinting
from typing import List, Tuple

# For image preprocessing
import numpy as np

# For accessing the dataset
from cell_counter.import_dataset import get_dataset_info, load_images_from_dataframe

#(get_dataset_info, get_dataset_info), (load_images_from_dataframe, load_images_from_dataframe) = tf.cell_counter.import_dataset
#get_dataset_info, load_images_from_dataframe = get_dataset_info/255.0 , load_images_from_dataframe/255.0

# For creating and using CNN
import tensorflow as tf

#resnet imports
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


#module from resnet git
def res_identity(x, filters): 

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input 
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x


#module from resnet git
def res_conv(x, s, filters):
  '''
  here the input size changes, when it goes via conv blocks
  so the skip connection uses a projection (conv layer) matrix
  ''' 
  x_skip = x
  f1, f2 = filters

  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)

  # shortcut 
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add 
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x

#module from resnet git
def resnet50():

  preprocessed_image_shape=(128,128,1)
  input_im = Input(shape=preprocessed_image_shape)
  x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = AveragePooling2D((2, 2), padding='same')(x)

  x = Flatten()(x)
  #x = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x) #multi-class
  x = Dense(64)(x)
  x = Dense(1, activation='relu')(x)
  # define the model 

  model = Model(inputs=input_im, outputs=x, name='Resnet50')
  #model = Model(inputs=preprocessed_image_shape, outputs=x, name='Resnet50')

  return model

def resnet_preprocess_data(
    path: str = None, num: int = 2500
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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
    df = df[df["blur"] == 1]

    # Randomly select 'num' from the remaining images, without replacement
    df = df.sample(n=num, replace=False)

    # Return images and labels from dataframe
    (training_images, training_labels), (
        testing_images,
        testing_labels,
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

def compile_resnet(model):
    """
    Compiles the resnetmodel.
    Parameters:
    model: The resnetto be compiled.
    """
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])


def run_resnet(
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
    Runs the resnetmodel.
    Parameters:
    model: The resnetmodel to run.
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
    ) = resnet_preprocess_data(path=path, num=image_number)
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

### Define some Callbacks
def lrdecay(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr
  # if epoch < 40:
  #   return 0.01
  # else:
  #   return 0.01 * np.math.exp(0.03 * (40 - epoch))
lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay  

def evaluate_model(model, name):
    import matplotlib.pyplot as plt
    path='C:\\Users\\User\\Documents\\BBC005Data\\BBBC005_v1_images\\'
    plt.plot(training_hist.history["mse"], label="mse")
    plt.plot(training_hist.history["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.savefig(name+'.png')

    # Test against blurless images
    df = get_dataset_info(path)
    df = df[df['blur']==1]
    df = df.sample(n=50, replace=False)

    print("Evaluating against blurless images...")
    (_,_),(test_im, test_lab) = resnet_preprocess_data(df=df, split=1)
    results = model.evaluate(test_im, test_lab, batch_size=1)

    # Test against random images
    df = get_dataset_info()
    df = df.sample(n=50, replace=False)

    print("Evaluating against random images...")
    (_,_),(test_im, test_lab) = resnet_preprocess_data(df=df, split=1)
    results = model.evaluate(test_im, test_lab, batch_size=1)

if __name__ == "__main__":
    
    model = resnet50()
    
    compile_resnet(model)

    training_hist, _ = run_resnet(
        model, epochs=10, image_number=250, path='C:\\Users\\User\\Documents\\BBC005Data\\BBBC005_v1_images\\',validation_split=0.1, checkpointing=False
    )
    
    '''
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('High training size, no blur')
    model = resnet50()
    compile_resnet(model)
    training_hist, _ = run_resnet(
        model, epochs=10,path='C:\\Users\\User\\Documents\\BBC005Data\\BBBC005_v1_images\\', image_number=1000, validation_split=0.1, checkpointing=False
    )
    '''
    evaluate_model(model, 'resnet_h_n')
    
    
    
    
    
    
    
    
    
    
    print(model.summary())
    
    import matplotlib.pyplot as plt

    
    plt.plot(training_hist.history["mse"], label="mse")
    plt.plot(training_hist.history["val_mse"], label="val_mse")
    plt.title("ResNet")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.show()

    
                       