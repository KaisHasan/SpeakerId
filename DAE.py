import tensorflow as tf
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.data import Dataset
import numpy as np
import gc


def train(unlabeled_data: np.ndarray,
          epochs: int = 20,
          batch_size: int = 8,
          learning_rate: float = 0.001,
          model=None) -> tf.keras.Sequential:
    """Train the autoencoder.

    Args:
        unlabeled_data (np.ndarray): Each row is a training example.
        epochs (int, optional): Defaults to 20.
        batch_size (int, optional):  Defaults to 8.
        learning_rate (float, optional):  Defaults to 0.001.
        model (tf.keras.Sequential, optional): pass this if you want to
        train an existing model otherwise a new default model will be created.
        Defaults to None.

    Returns:
        tf.keras.Sequential: the model after training.
    """

    # number of examples in the training set
    n = int(0.9 * unlabeled_data.shape[0])
    x_train = unlabeled_data[:n]  # get the training set
    # add standard gaussian noise to each example
    x_train_noisy = unlabeled_data[:n] + 0.2*np.random.normal(
        loc=0.0, scale=1.0, size=unlabeled_data[:n].shape
    )
    # masking the spectrogram in the frequency domain
    prob = np.random.rand(x_train.shape[0], 512)
    mask = np.zeros(x_train.shape)
    mask[prob > 0.01] = 1
    x_train_noisy = x_train_noisy * mask
    x_val = unlabeled_data[n:]  # get the validation set
    # default model parameters
    alpha = 0.7  # for leakyRelu
    reg_lambda = 1e-4  # for regularization

    if model is None:
        model = tf.keras.Sequential(
            [
                BatchNormalization(),
                # (512, 301)
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(2, 2),
                    kernel_regularizer=l2(reg_lambda),
                ),
                LeakyReLU(alpha),
                # (511, 300)
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(3, 3),
                    kernel_regularizer=l2(reg_lambda),
                ),
                LeakyReLU(alpha),
                # (509, 298)
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(4, 3),
                    kernel_regularizer=l2(reg_lambda),
                ),
                LeakyReLU(alpha),
                # (506, 296)
                tf.keras.layers.MaxPool2D((2, 2)),
                # (253, 148)


                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(5, 5),
                    kernel_regularizer=l2(reg_lambda),
                ),
                LeakyReLU(alpha),
                # (249, 144)
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(6, 5),
                    kernel_regularizer=l2(reg_lambda),
                ),
                LeakyReLU(alpha),
                # (244, 140)
                tf.keras.layers.MaxPool2D((2, 2)),
                # (122, 70)


                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(7, 7),
                    kernel_regularizer=l2(reg_lambda),
                ),
                LeakyReLU(alpha),
                # (116, 64)
                tf.keras.layers.MaxPool2D((2, 2)),
                # (58, 32)


                ##################################################
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2DTranspose(
                    filters=16,
                    kernel_size=(7, 7),
                    kernel_regularizer=l2(reg_lambda)
                ),
                LeakyReLU(alpha),

                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2DTranspose(
                    filters=16,
                    kernel_size=(6, 5),
                    kernel_regularizer=l2(reg_lambda)
                ),
                LeakyReLU(alpha),
                tf.keras.layers.Conv2DTranspose(
                    filters=16,
                    kernel_size=(5, 5),
                    kernel_regularizer=l2(reg_lambda)
                ),
                LeakyReLU(alpha),
                tf.keras.layers.UpSampling2D((2, 2)),

                tf.keras.layers.Conv2DTranspose(
                    filters=8,
                    kernel_size=(4, 3),
                    kernel_regularizer=l2(reg_lambda)
                ),
                LeakyReLU(alpha),
                tf.keras.layers.Conv2DTranspose(
                    filters=8,
                    kernel_size=(3, 3),
                    kernel_regularizer=l2(reg_lambda)
                ),
                LeakyReLU(alpha),
                tf.keras.layers.Conv2DTranspose(
                    filters=8,
                    kernel_size=(2, 2),
                    kernel_regularizer=l2(reg_lambda)
                ),
                LeakyReLU(alpha),
                ##################################################

                tf.keras.layers.Conv2D(filters=1,
                                       kernel_size=(1, 1)),
                LeakyReLU(alpha),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.losses.mean_squared_error
        )

    print('Start training the DAE coder...')

    # convert dataset to more efficient type
    train_data = Dataset.from_tensor_slices((x_train_noisy, x_train))
    val_data = Dataset.from_tensor_slices((x_val, x_val))

    model.fit(
        train_data.batch(batch_size),
        validation_data=val_data.batch(batch_size),
        epochs=epochs
    )

    # we used the following to solve some memory issues
    # due to small RAMs capacity.
    del unlabeled_data
    del x_val
    del x_train
    del train_data
    del val_data
    gc.collect()
    return model
