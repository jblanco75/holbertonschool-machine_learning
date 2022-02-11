#!/usr/bin/env python3
"""
Function that builds the DenseNet-121 architecture as
described in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    You may use:
      dense_block = __import__('5-dense_block').dense_block
      transition_layer = __import__('6-transition_layer').transition_layer
    Returns: the keras model
    """
    w = K.initializers.HeNormal()
    inputs = K.Input(shape=(224, 224, 3))
    n_filters = growth_rate * 2
    BN = K.layers.BatchNormalization()(inputs)
    act_BN = K.layers.Activation('relu')(BN)
    conv_1 = K.layers.Conv2D(filters=n_filters,
                             kernel_size=(7, 7),
                             padding='same',
                             strides=2,
                             kernel_initializer=w)(act_BN)
    pool_1 = K.layers.MaxPool2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(conv_1)
    X1, n_filters = dense_block(pool_1, n_filters, growth_rate, 6)
    Tr_layer1, n_filters = transition_layer(X1, n_filters, compression)
    X2, n_filters = dense_block(Tr_layer1, n_filters, growth_rate, 12)
    Tr_layer2, n_filters = transition_layer(X2, n_filters, compression)
    X3, n_filters = dense_block(Tr_layer2, n_filters, growth_rate, 24)
    Tr_layer3, n_filters = transition_layer(X3, n_filters, compression)
    X4, n_filters = dense_block(Tr_layer3, n_filters, growth_rate, 16)

    pool_2 = K.layers.AveragePooling2D(pool_size=7)(X4)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=w)(pool_2)
    return K.Model(inputs=inputs, outputs=softmax)
