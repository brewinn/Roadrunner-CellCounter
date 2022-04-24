#For building and running the model
import tensorflow as tf

#For importing the dataset
from cell_counter.import_dataset import load_synthetic_dataset

def fcrn_preprocess_data( path: str = None, num: int = 2500):
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
    (training_images, training_labels), (
        testing_images,
        testing_labels,
    ) = load_synthetic_dataset(path=path, num=num, resolution=(128, 128))

    scale = 1 / float(255)
    for index, image in enumerate(training_images):
        training_images[index] = image * scale
    for index, image in enumerate(testing_images):
        testing_images[index] = image * scale

    return (training_images, training_labels), (
        testing_images,
        testing_labels,
    )

def build_fcrn():
    """
    Returns a FCRN model for use on a preprocessed dataset.
    """

    preprocessed_image_shape = (128, 128, 1)

    from tensorflow.keras import Model, layers

    def _conv_bn_relu(filters, kernel_size, subsample = (1,1), activation='relu', weight_decay = 1e-5):
        def f(input_):
            conv_a = layers.Conv2D(
                    filters, kernel_size, strides = subsample, 
                    activation=activation,
                    padding='same',
                    use_bias=False,
                    input_shape=preprocessed_image_shape, 
                    kernel_initializer = 'orthogonal',
                    kernel_regularizer = tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer = tf.keras.regularizers.L2(weight_decay),
                )(input_)
            norm_a = layers.BatchNormalization()(conv_a)
            act_a = layers.Activation(activation = activation)(norm_a)
            return act_a
        return f

    def _conv_bn_relu_x2(filters, kernel_size, subsample = (1,1), activation='relu', weight_decay = 1e-5):
        def f(input_):
            conv_a = layers.Conv2D(
                    filters, kernel_size, strides = subsample, 
                    activation=activation,
                    padding='same',
                    use_bias=False,
                    input_shape=preprocessed_image_shape, 
                    kernel_initializer = 'orthogonal',
                    kernel_regularizer = tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer = tf.keras.regularizers.L2(weight_decay),
                )(input_)
            norm_a = layers.BatchNormalization()(conv_a)
            act_a = layers.Activation(activation = activation)(norm_a)
            conv_b = layers.Conv2D(
                    filters, kernel_size, strides = subsample, 
                    activation=activation,
                    padding='same',
                    use_bias=False,
                    input_shape=preprocessed_image_shape, 
                    kernel_initializer = 'orthogonal',
                    kernel_regularizer = tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer = tf.keras.regularizers.L2(weight_decay),
                )(act_a)
            norm_b = layers.BatchNormalization()(conv_b)
            act_b = layers.Activation(activation = activation)(norm_b)
            return act_b
        return f

    def fcrn_base(input_):
        block1 = _conv_bn_relu_x2(32,(3,3))(input_)
        pool1 = layers.MaxPooling2D(pool_size=(2,2))(block1)
        # =========================================================================
        block2 = _conv_bn_relu_x2(64,(3,3))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)
        # =========================================================================
        block3 = _conv_bn_relu_x2(128,(3,3))(pool2)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)
        # =========================================================================
        block4 = _conv_bn_relu(512,(3,3))(pool3)
        # =========================================================================
        up5 = layers.UpSampling2D(size=(2, 2))(block4)
        block5 = _conv_bn_relu_x2(128,(3,3))(up5)
        # =========================================================================
        up6 = layers.UpSampling2D(size=(2, 2))(block5)
        block6 = _conv_bn_relu_x2(64,(3,3))(up6)
        # =========================================================================
        up7 = layers.UpSampling2D(size=(2, 2))(block6)
        block7 = _conv_bn_relu_x2(32,(3,3))(up7)
        return block7
    
    input_ = layers.Input(shape = preprocessed_image_shape)

    act_ = fcrn_base(input_)

    density_pred =  layers.Conv2D(1, (1,1), use_bias=False,
            activation='linear', kernel_initializer='orthogonal', name='pred',
            padding='same')(act_)

    model = Model(inputs = input_, outputs=density_pred)

    return model

def compile_fcrn(model):
    opt = tf.keras.optimizers.SGD()
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])

def run_fcrn(
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
    Runs the FCRN model.

    Parameters:
    model: The FCRN model to run.
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
    ) = fcrn_preprocess_data(path=path, num=image_number)
    if checkpointing:
        if not checkpoint_path:
            import os

            counter_dir = os.path.dirname(__file__)
            checkpoint_path = counter_dir + "/../resources/checkpoints/fcrn_cp.ckpt"

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
    model = build_fcrn()
    compile_fcrn(model)

    training_hist, _ = run_fcrn(
        model, epochs=10, image_number=250, validation_split=0.1, checkpointing=False
    )

    import matplotlib.pyplot as plt

    plt.plot(training_hist.history["mse"], label="mse")
    plt.plot(training_hist.history["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.show()
