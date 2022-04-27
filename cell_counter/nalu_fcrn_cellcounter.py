# For building and running the model
import tensorflow as tf
import tensorflow.math as math
from tensorflow.linalg import matmul as dot

# For importing the dataset
from cell_counter.import_dataset import load_synthetic_dataset


class NALU(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        MW_initializer="glorot_uniform",
        G_initializer="glorot_uniform",
        mode="NALU",
        **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.MW_initializer = tf.keras.initializers.get(MW_initializer)
        self.G_initializer = tf.keras.initializers.get(G_initializer)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(
            shape=(input_dim, self.units), initializer=self.MW_initializer, name="W_hat"
        )
        self.M_hat = self.add_weight(
            shape=(input_dim, self.units), initializer=self.MW_initializer, name="M_hat"
        )
        if self.mode == "NALU":
            self.G = self.add_weight(
                shape=(input_dim, self.units), initializer=self.G_initializer, name="G"
            )
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        W = math.tanh(self.W_hat) * math.sigmoid(self.M_hat)
        a = dot(inputs, W)
        if self.mode == "NAC":
            output = a
        elif self.mode == "NALU":
            m = math.exp(dot(math.log(math.abs(inputs) + 1e-7), W))
            g = math.sigmoid(dot(math.abs(inputs), self.G))
            output = g * a + (1 - g) * m
        else:
            raise ValueError("Valid modes: 'NAC', 'NALU'.")
        return output


def nalu_fcrn_preprocess_data(path: str = None, num: int = 2500):
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


def build_nalu_fcrn():
    """
    Returns a FCRN model for use on a preprocessed dataset.
    """

    preprocessed_image_shape = (128, 128, 1)

    from tensorflow.keras import Model, layers, initializers

    def _conv_bn_relu(
        filters, kernel_size, subsample=(1, 1), activation="relu", weight_decay=1e-5
    ):
        def f(input_):
            conv_a = layers.Conv2D(
                filters,
                kernel_size,
                strides=subsample,
                activation=activation,
                padding="same",
                use_bias=False,
                input_shape=preprocessed_image_shape,
                kernel_initializer="orthogonal",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                bias_regularizer=tf.keras.regularizers.L2(weight_decay),
            )(input_)
            norm_a = layers.BatchNormalization()(conv_a)
            act_a = layers.Activation(activation=activation)(norm_a)
            return act_a

        return f

    def _conv_bn_relu_x2(
        filters, kernel_size, subsample=(1, 1), activation="relu", weight_decay=1e-5
    ):
        def f(input_):
            conv_a = layers.Conv2D(
                filters,
                kernel_size,
                strides=subsample,
                activation=activation,
                padding="same",
                use_bias=False,
                input_shape=preprocessed_image_shape,
                kernel_initializer="orthogonal",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                bias_regularizer=tf.keras.regularizers.L2(weight_decay),
            )(input_)
            norm_a = layers.BatchNormalization()(conv_a)
            act_a = layers.Activation(activation=activation)(norm_a)
            conv_b = layers.Conv2D(
                filters,
                kernel_size,
                strides=subsample,
                activation=activation,
                padding="same",
                use_bias=False,
                input_shape=preprocessed_image_shape,
                kernel_initializer="orthogonal",
                kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                bias_regularizer=tf.keras.regularizers.L2(weight_decay),
            )(act_a)
            norm_b = layers.BatchNormalization()(conv_b)
            act_b = layers.Activation(activation=activation)(norm_b)
            return act_b

        return f

    # Original base, as used in the paper. Outputs heatmap.
    def fcrn_nalu(input_):
        block1 = _conv_bn_relu_x2(32, (3, 3))(input_)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(block1)
        nal1 = NALU(
            32,
            mode="NAC",
            MW_initializer=initializers.RandomNormal(stddev=1),
            G_initializer=initializers.Constant(10),
        )(block1)
        # =========================================================================
        block2 = _conv_bn_relu_x2(64, (3, 3))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)
        nal2 = NALU(
            64,
            mode="NAC",
            MW_initializer=initializers.RandomNormal(stddev=1),
            G_initializer=initializers.Constant(10),
        )(block2)
        # =========================================================================
        block3 = _conv_bn_relu_x2(128, (3, 3))(pool2)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)
        nal3 = NALU(
            128,
            mode="NAC",
            MW_initializer=initializers.RandomNormal(stddev=1),
            G_initializer=initializers.Constant(10),
        )(block3)
        # =========================================================================
        block4 = _conv_bn_relu(512, (3, 3))(pool3)
        # =========================================================================
        up5 = layers.UpSampling2D(size=(2, 2))(block4)
        block5 = layers.Concatenate()([_conv_bn_relu_x2(128, (3, 3))(up5), nal3])
        block5 = _conv_bn_relu_x2(128, (3, 3))(up5)
        # =========================================================================
        up6 = layers.UpSampling2D(size=(2, 2))(block5)
        block6 = layers.Concatenate()([_conv_bn_relu_x2(64, (3, 3))(up6), nal2])
        block6 = _conv_bn_relu_x2(64, (3, 3))(up6)
        # =========================================================================
        up7 = layers.UpSampling2D(size=(2, 2))(block6)
        block7 = layers.Concatenate()([_conv_bn_relu_x2(32, (3, 3))(up7), nal1])
        block7 = _conv_bn_relu_x2(32, (3, 3))(up7)
        return block7

    # Modified base, outputs single prediction for cellcount
    def modified_fcrn_nalu(input_):
        block1 = _conv_bn_relu_x2(32, (3, 3))(input_)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(block1)
        nal1 = NALU(
            32,
            mode="NAC",
            MW_initializer=initializers.RandomNormal(stddev=1),
            G_initializer=initializers.Constant(10),
        )(block1)
        # =========================================================================
        block2 = _conv_bn_relu_x2(64, (3, 3))(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)
        nal2 = NALU(
            64,
            mode="NAC",
            MW_initializer=initializers.RandomNormal(stddev=1),
            G_initializer=initializers.Constant(10),
        )(block2)
        # =========================================================================
        block3 = _conv_bn_relu_x2(128, (3, 3))(pool2)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)
        nal3 = NALU(
            128,
            mode="NAC",
            MW_initializer=initializers.RandomNormal(stddev=1),
            G_initializer=initializers.Constant(10),
        )(block3)
        # =========================================================================
        block4 = _conv_bn_relu(512, (3, 3))(pool3)
        # =========================================================================
        up5 = layers.UpSampling2D(size=(2, 2))(block4)
        block5 = layers.Concatenate()([_conv_bn_relu_x2(128, (3, 3))(up5), nal3])
        block5 = _conv_bn_relu_x2(128, (3, 3))(up5)
        # =========================================================================
        # Append simple cnn to reduce the final output to appropriate dimensions
        cnn_filter = 16
        cnn_pool_size = 2
        kernal = (3, 3)
        dropout_rate = 0.2

        outputs_ = layers.Conv2D(cnn_filter, kernal, activation="relu")(block5)
        outputs_ = layers.MaxPooling2D(pool_size=cnn_pool_size)(outputs_)
        outputs_ = layers.Dropout(dropout_rate)(outputs_)
        outputs_ = layers.Conv2D(2 * cnn_filter, kernal, activation="relu")(outputs_)
        outputs_ = layers.MaxPooling2D(pool_size=cnn_pool_size)(outputs_)
        outputs_ = layers.Conv2D(2 * cnn_filter, kernal, activation="relu")(outputs_)

        outputs_ = layers.Flatten()(outputs_)
        outputs_ = layers.Dense(64)(outputs_)
        outputs_ = layers.Dense(1, activation="relu")(outputs_)
        return outputs_

    input_ = layers.Input(shape=preprocessed_image_shape)

    outputs_ = modified_fcrn_nalu(input_)

    # act_ = fcrn_nalu(input_)

    # density_pred =  layers.Conv2D(1, (1,1), use_bias=False, activation='linear', kernel_initializer='orthogonal', name='pred', padding='same')(act_)

    model = Model(inputs=input_, outputs=outputs_)

    return model


def compile_nalu_fcrn(model):
    # opt = tf.keras.optimizers.SGD()
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])


def run_nalu_fcrn(
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
    Runs the NALU FCRN model.

    Parameters:
    model: The NALU FCRN model to run.
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
    ) = nalu_fcrn_preprocess_data(path=path, num=image_number)
    if checkpointing:
        if not checkpoint_path:
            import os

            counter_dir = os.path.dirname(__file__)
            checkpoint_path = (
                counter_dir + "/../resources/checkpoints/nalu_fcrn_cp.ckpt"
            )

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
    model = build_nalu_fcrn()
    compile_nalu_fcrn(model)

    training_hist, _ = run_nalu_fcrn(
        model, epochs=10, path='C:\\Users\\User\\Documents\\BBC005Data\\BBBC005_v1_images\\', image_number=250, validation_split=0.1, checkpointing=False
    )

    import matplotlib.pyplot as plt

    plt.plot(training_hist.history["mse"], label="mse")
    plt.plot(training_hist.history["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="upper right")
    plt.show()
