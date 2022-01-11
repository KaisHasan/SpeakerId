import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
import gc


def record_read(record_bytes):
    """Read the data stored in a tfrecords files.

    Args:
        record_bytes (tf.data.TFRecordDataset): the data read from tfrecords
        files by using tf.data.TFRecordDataset

    Returns:
        [type]: [description]
    """
    parsed_features = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"x": tf.io.FixedLenFeature([58*32*16], dtype=tf.float32),
         "y": tf.io.FixedLenFeature([1251], dtype=tf.float32)}
    )
    x = parsed_features['x']
    x = tf.reshape(x, (58, 32, 16))
    y = parsed_features['y']
    return x, y


# Put the files names in a list
train_files_name = []
val_files_name = []
for j in range(7):
    for i in range(18):
        train_files_name.append(f'dataset/vox_part{j}.{i+1}.tfrecords')
    val_files_name.append(f'dataset/vox_part{j}.19.tfrecords')

# Read the files and uncompress them
train_dataset = tf.data.TFRecordDataset(
    filenames=train_files_name, compression_type='GZIP',
    num_parallel_reads=6
)

# Convert the file content to the required form from the serialized one
train_dataset = train_dataset.map(
    record_read, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
train_dataset = train_dataset.shuffle((1 << 11))  # shuffle the data
# Slice the dataset into batches of size 800
train_dataset = train_dataset.batch(
    (800), num_parallel_calls=tf.data.AUTOTUNE,
    drop_remainder=True  # to ensure that each batch have same size
)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# Do the same as above but for the validation set
val_dataset = tf.data.TFRecordDataset(
    filenames=val_files_name, compression_type='GZIP',
    num_parallel_reads=6
)

val_dataset = val_dataset.map(
    record_read, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
val_dataset = val_dataset.batch(
    (800), num_parallel_calls=tf.data.AUTOTUNE,
    drop_remainder=True
)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


reg_lambda = 1e-2  # for l2 regularizer
rate = 0.6  # the rate of dropout
model = tf.keras.Sequential(
    [
        # 58, 32, 16
        # ** 1st Conv Block **
        Conv2D(
            filters=16,
            kernel_size=(7, 1),
            strides=(2, 1),
            kernel_regularizer=l2(reg_lambda)
        ),
        ReLU(),
        BatchNormalization(),
        Conv2D(
            filters=32,
            kernel_size=(9, 1),
            strides=(2, 1),
            kernel_regularizer=l2(reg_lambda)
        ),
        ReLU(),
        BatchNormalization(),
        AveragePooling2D(pool_size=(1, 5), strides=(1, 3)),
        # 9, 10, 64
        Dropout(0.2),


        Flatten(),
        # 5408

        # ** Dense Block **
        Dense(
          units=1024,
          kernel_regularizer=l2(reg_lambda)
        ),
        ReLU(),
        BatchNormalization(),
        Dropout(0.6),
        Dense(
          units=1024,
          kernel_regularizer=l2(reg_lambda)
        ),
        ReLU(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(
          units=1251,
          kernel_regularizer=l2(reg_lambda)
        ),
        Softmax()
    ]
)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10_000,
    decay_rate=0.6
)
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy']
)

mc = tf.keras.callbacks.ModelCheckpoint(
    'ModelCheckpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=0, save_best_only=False,
    save_weights_only=False, mode='min',
    save_freq=1610
)

tb = tf.keras.callbacks.TensorBoard()

model.fit(
  train_dataset,
  epochs=10_000,
  validation_data=val_dataset,
  verbose=1,
  callbacks=[mc, tb]
)
