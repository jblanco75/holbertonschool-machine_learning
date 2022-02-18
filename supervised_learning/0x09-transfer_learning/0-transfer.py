#!/usr/bin/env python3
"""
Python script that trains a convolutional
neural network to classify the CIFAR 10 dataset
"""


import tensorflow as tf
import tensorflow.keras as K


def preprocess_data(X, Y):
    """Data preprocessing"""
    X_p = K.applications.xception.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    clf_model = K.applications.Xception(include_top=False,
                                        weights="imagenet",
                                        input_shape=(299, 299, 3))

    clf_model.summary()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inp = K.Input(shape=(32, 32, 3))
    inp_resized = K.layers.Lambda(lambda X:
                                  tf.image.resize(X, (299, 299)))(inp)

    X = clf_model(inp_resized, training=False)
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(500, activation='relu')(X)
    X = K.layers.Dropout(0.3)(X)
    outputs = K.layers.Dense(10, activation='softmax')(X)

    model = K.Model(inp, outputs)

    clf_model.trainable = False
    optimizer = K.optimizers.Adam(learning_rate=0.00001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=300, epochs=4, verbose=1)

    model.save('cifar10.h5')
