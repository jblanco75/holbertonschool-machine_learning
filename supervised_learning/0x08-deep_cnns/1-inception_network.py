#!/usr/bin/env python3
"""
Function that builds an inception block as
described in Going Deeper with Convolutions (2014)
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block
    should use a rectified linear activation (ReLU)
    Returns: the keras model
    """
    w = K.initializers.HeNormal()
    inputs = K.layers.Input(shape=(224, 224, 3))
    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             padding='same',
                             activation='relu',
                             strides=2,
                             kernel_initializer=w)(inputs)
    pool_1 = K.layers.MaxPool2D(pool_size=3, strides=2,
                                padding='same')(conv_1)
    conv_2 = K.layers.Conv2D(filters=192,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu',
                             strides=1,
                             kernel_initializer=w)(pool_1)
    pool_2 = K.layers.MaxPool2D(pool_size=3, strides=2,
                                padding='same')(conv_2)
    inblock_3a = inception_block(pool_2, [64, 96, 128, 16, 32, 32])
    inblock_3b = inception_block(inblock_3a, [128, 128, 192, 32, 96, 64])
    pool_3 = K.layers.MaxPool2D(pool_size=3, strides=2,
                                padding='same')(inblock_3b)
    inblock_4a = inception_block(pool_3, [192, 96, 208, 16, 48, 64])
    inblock_4b = inception_block(inblock_4a, [160, 112, 224, 24, 64, 64])
    inblock_4c = inception_block(inblock_4b, [128, 128, 256, 24, 64, 64])
    inblock_4d = inception_block(inblock_4c, [112, 144, 288, 32, 64, 64])
    inblock_4e = inception_block(inblock_4d, [256, 160, 320, 32, 128, 128])
    pool_4 = K.layers.MaxPool2D(pool_size=3, strides=2,
                                padding='same')(inblock_4e)
    inblock_5a = inception_block(pool_4, [256, 160, 320, 32, 128, 128])
    inblock_5b = inception_block(inblock_5a, [384, 192, 384, 48, 128, 128])
    pool_4 = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                       padding='valid')(inblock_5b)
    drop_out_layer = K.layers.Dropout(0.4)(pool_4)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=w)(drop_out_layer)
    mod = K.Model(inputs=inputs, outputs=softmax)
    return mod
