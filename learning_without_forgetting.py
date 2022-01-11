import os
import tensorflow as tf
import numpy as np


class LearningWithoutForgetting(object):
    """Learning without Forgetting functionality.

    Note: We are using a custom loss function
    so remember that when you load a saved model.
    """

    @staticmethod
    def add_n_class(model: tf.keras.Model,
                    x_train: np.array,
                    y_train: np.array,
                    num_classes: int = 2,
                    new_lr: float = 0.001,
                    all_lr: float = 0.0001,
                    batch_size: int = 4,
                    new_epochs: int = 1,
                    all_epochs: int = 1) -> tf.keras.Model:
        """Adding a new unit for each class to the output
           layers to classify one more class.
           it includes two phases:
            First: train the new units only, while freezing all other layers.
            Second: train the whole network.
           Note: The number of classes must be greater than 1.

        Args:
            model (tf.keras.Model): A model to add to it.
            x_train (np.array): Training set for the new classes only,
            its shape must be (num_examples, features)
            y_train (np.array): Labels for the training set,
            its shape must be (None, num_classes)
            num_classes (int): Number of classes to add. Defaults to 2.
            new_lr (float, optional): Learning rate for first phase.
            Defaults to 0.001.
            all_lr (float, optional): Learning rate for second phase.
            Defaults to 0.0001.
            batch_size (int, optional): Batch size for both training phases.
            Defaults to 4.
            new_epochs (int, optional): Number of epochs for first phase.
            Defaults to 1.
            all_epochs (int, optional): Number of epochs for second phase.
            Defaults to 1.

        Returns:
            tf.keras.Model: The same given model after adding one more unit
            for each class to the output layer and do the two training phases.
        """
        # get output of old tasks on new data
        y_o = model(x_train)
        # add a new unit for each new class
        # and do the warm up step
        temp = LearningWithoutForgetting._add_new_units(
            model, x_train, y_train,
            num_classes,
            new_lr=new_lr,
            batch_size=batch_size,
            new_epochs=new_epochs
        )
        # expand y_train to contain the output of
        # old tasks on the new task
        expanded_y_train = np.append(y_o, y_train, axis=-1)
        assert expanded_y_train.shape == (y_train.shape[0],
                                          y_o.shape[1]+y_train.shape[1])
        # train the whole model
        all_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=all_lr,
            decay_steps=100,
            decay_rate=0.6
        )
        temp.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=all_lr_schedule),
            loss=LearningWithoutForgetting._loss_function(num_classes),
            metrics=['accuracy']
        )
        print('Start training the whole model...')
        temp.fit(
            x_train, expanded_y_train,
            batch_size=batch_size,
            epochs=all_epochs
        )
        return temp

    @staticmethod
    def _add_new_units(base_model: tf.keras.Model,
                       x_train: np.array,
                       y_train: np.array,
                       num_classes: int,
                       new_lr: float,
                       batch_size: int,
                       new_epochs: int,
                       last_hidden_layer_number: int = -6,
                       output_layer_number: int = -2,
                       output_block_size: int = 3) -> tf.keras.Model:
        """Add the new units and make a merged model ready for training.

        Args:
            base_model (tf.keras.Model): The model we want to expand
            x_train (np.array): Training examples
            y_train (np.array): Training examples' labels
            num_classes (int): The number of new classes to add
            new_lr (float): The learning rate for the training of
            the new classes
            batch_size (int):
            new_epochs (int):
            last_hidden_layer_number (int, optional): The number of the last
            hidden layer in the sequential model.
            Note that you can pass a negative number for counting from the end.
            Defaults to -6.
            output_layer_number (int, optional): The number of the output
            layer in the sequential model.
            Note that you can pass a negative number for counting from the end.
            Defaults to -2.
            output_block_size (int, optional): The number of layers that are
            related to output layer, for example the output layer and a dropout
            and a softmax so we have size 3.
            Defaults to 3.

        Returns:
            tf.keras.Model: The merged model ready for training.
        """
        # num of units in the last hidden layer
        n0 = base_model.layers[last_hidden_layer_number].units
        # num of units in the output layer
        n1 = base_model.layers[output_layer_number].units

        # get the weights of the old output layer
        old_w = base_model.layers[-2].get_weights()[0]
        old_b = base_model.layers[-2].get_weights()[1]
        # Check parameters' value
        nans = np.count_nonzero(np.isnan(old_w))
        assert nans == 0
        nans = np.count_nonzero(np.isnan(old_b))
        assert nans == 0
        # remove the old output layer
        for i in range(output_block_size):
            base_model.pop()
        # get the features
        new_x_train = base_model.predict(x_train)
        # create the new output layer
        # TODO: make this more dynamic for user.
        new_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(
                    units=num_classes,
                    kernel_regularizer='l2'
                ),
                tf.keras.layers.Softmax()
            ]
        )
        new_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=new_lr,
            decay_steps=100,
            decay_rate=0.6
        )
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr_schedule),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        print('Start training the freezed model...')
        new_model.fit(
            new_x_train, y_train, batch_size=batch_size,
            epochs=new_epochs
        )

        # merge the two output layers in one layer
        merged_model = base_model
        new_w = np.append(
            old_w,
            new_model.layers[-2].get_weights()[0],
            axis=-1
        )
        # Check parameters' value and shape
        nans = np.count_nonzero(np.isnan(new_w))
        assert nans == 0
        assert new_w.shape == (n0, n1 + num_classes)

        new_b = np.append(old_b, new_model.layers[-2].get_weights()[1])
        # Check parameters' value and shape
        nans = np.count_nonzero(np.isnan(new_b))
        assert nans == 0
        assert new_b.shape == (n1 + num_classes,)

        # Add the new merged output block
        merged_model.add(tf.keras.layers.Dropout(0.6))
        merged_model.add(tf.keras.layers.Dense(units=n1+num_classes,
                                               kernel_regularizer='l2',
                                               input_shape=(n0,)))
        merged_model.add(tf.keras.layers.Softmax())
        # Set the weights to the weights taken from both
        # old and new output layers
        merged_model.layers[-2].set_weights([new_w, new_b])
        return merged_model

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _loss_function(num_classes):
        @tf.autograph.experimental.do_not_convert
        def _loss(y_true, y_pred):
            # modify probablities for Knowledge Distillation loss
            # we do this for old tasks only
            old_y_true = tf.math.pow(y_true[:, :-num_classes],
                                     tf.constant(0.5))
            old_y_true = tf.math.divide(old_y_true,
                                        tf.math.reduce_sum(old_y_true))
            old_y_pred = tf.math.pow(y_pred[:, :-num_classes],
                                     tf.constant(0.5))
            old_y_pred = tf.math.divide(old_y_pred,
                                        tf.math.reduce_sum(old_y_pred))
            # Define the loss that we will used for new and old tasks
            bce = tf.keras.losses.BinaryCrossentropy()
            # compute the loss on old tasks
            old_loss = bce(old_y_true, old_y_pred)
            # compute the loss on new task
            new_loss = bce(y_true[:, -num_classes:], y_pred[:, -num_classes:])
            # convert all tensors to float64
            old_loss = tf.cast(old_loss, dtype=tf.float64)
            new_loss = tf.cast(new_loss, dtype=tf.float64)
            return old_loss + new_loss

        return _loss

    @staticmethod
    def _deep_copy_model(model: tf.keras.Model):
        """Copy layers from one model to another as new layers
        but with the same weights.
        Note that copying involves only some parameters as follows:
            dense: units, activation, kernel_regularizer.
            conv1d: filters, kernel_size, strides, activation,
                    kernel_regularizer.
            conv2d: filters, kernel_size, strides, activation,
                    kernel_regularizer.
            max_pooling1d: pool_size, strides.
            max_pooling2d: pool_size, strides.

        Args:
            model (tf.keras.Model): the model to copy

        Raises:
            TypeError: if there is a layer with unsupported type.
            supported types are: 'dense', 'conv1d', 'conv2d',
            'max_pooling1d', 'max_pooling2d'

        Returns:
            tf.keras.Model: the copied model
        """
        temp = tf.keras.Sequential([])
        # copy layers
        for layer in model.layers:
            if layer.name.startswith('dense'):
                temp.add(tf.keras.layers.Dense(
                    units=layer.units,
                    activation=layer.activation,
                    kernel_regularizer=layer.kernel_regularizer
                ))
            elif layer.name.startswith('conv1d'):
                temp.add(tf.keras.layers.Conv1D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    activation=layer.activation,
                    kernel_regularizer=layer.kernel_regularizer
                ))
            elif layer.name.startswith('conv2d'):
                temp.add(tf.keras.layers.Conv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    activation=layer.activation,
                    kernel_regularizer=layer.kernel_regularizer
                ))
            elif layer.name.startswith('max_pooling1d'):
                temp.add(tf.keras.layers.MaxPool1D(
                    pool_size=layer.pool_size,
                    strides=layer.strides
                ))
            elif layer.name.startswith('max_pooling2d'):
                temp.add(tf.keras.layers.MaxPool2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides
                ))
            else:
                raise TypeError(f'{layer.name} is not supported!')
        # copy weights
        temp.build(model.input_shape)  # build to create weights
        for i, layer in enumerate(temp.layers):
            layer.set_weights(model.layers[i].get_weights())
        return temp


if __name__ == '__main__':
    # create the normal training set
    x_train = np.random.randn(100, 100)
    y_train = np.zeros((100, 10))
    for i in range(10):
        one_hot = np.zeros((1, 10))
        one_hot[0, i % 10] = 1
        y_train[i] = one_hot.copy()
    np.random.shuffle(y_train)

    # create the basic model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=100, activation='relu',
                                  kernel_regularizer='l2'),
            tf.keras.layers.Dense(units=128, activation='relu',
                                  kernel_regularizer='l2'),
            tf.keras.layers.Dense(units=64, activation='relu',
                                  kernel_regularizer='l2'),
            tf.keras.layers.Dense(units=128, activation='relu',
                                  kernel_regularizer='l2'),
            tf.keras.layers.Dense(units=10, activation='sigmoid',
                                  kernel_regularizer='l2'),
        ]
    )
    # train the model
    model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy())
    model.fit(x_train, y_train, batch_size=32, epochs=1)
    model.summary()

    # create training dataset for a new classes
    x_train = np.random.randn(30, 100)
    y_train = np.zeros((30, 3))
    y_train[:10, 0] = 1
    y_train[10:20, 1] = 1
    y_train[20:, 2] = 1
    np.random.shuffle(y_train)

    # pass the model to add the new classes
    model = LearningWithoutForgetting.add_n_class(
        model, x_train, y_train,
        num_classes=3,
        new_lr=0.01, all_lr=0.001,
        batch_size=2,
        new_epochs=5, all_epochs=100
    )
    model.summary()
    model.save('t.h5')

    # here we must handle problems that arise
    # from the custom function that we used
    # because we have intorduced an additional parameter
    # we can just load the model with compile=False.
    new_model = tf.keras.models.load_model('t.h5', compile=False)
    # create training dataset for a new classes
    x_train = np.random.randn(20, 100)
    y_train = np.zeros((20, 2))
    y_train[:10, 0] = 1
    y_train[10:, 1] = 1
    np.random.shuffle(y_train)

    # pass the model to add the new classes
    new_model = LearningWithoutForgetting.add_n_class(
        new_model, x_train, y_train,
        num_classes=2,
        new_lr=0.01, all_lr=0.001,
        batch_size=4,
        new_epochs=1, all_epochs=1
    )
    new_model.summary()
