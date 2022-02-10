#!/usr/bin/env python3
"""
Function that builds the ResNet-50 architecture as
described in Deep Residual Learning for Image Recognition (2015)
"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be
    followed by batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    You may use:
      identity_block = __import__('2-identity_block').identity_block
      projection_block = __import__('3-projection_block').projection_block
    Returns: the keras model
    """
    w = K.initializers.HeNormal()
    inputs = K.Input(shape=(224, 224, 3))
    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             padding='same',
                             kernel_initializer=w,
                             strides=(2, 2))(inputs)
    BN_1 = K.layers.BatchNormalization(axis=3)(conv_1)
    activ_1 = K.layers.Activation('relu')(BN_1)
    pool_1 = K.layers.MaxPool2D(pool_size=3,
                                strides=2,
                                padding='same')(activ_1)
    PB_1 = projection_block(pool_1, [64, 64, 256], s=1)
    ID_1 = identity_block(PB_1, [64, 64, 256])
    ID_2 = identity_block(ID_1, [64, 64, 256])
    PB_2 = projection_block(ID_2, [128, 128, 512])
    ID_3 = identity_block(PB_2, [128, 128, 512])
    ID_4 = identity_block(ID_3, [128, 128, 512])
    ID_5 = identity_block(ID_4, [128, 128, 512])
    PB_3 = projection_block(ID_5, [256, 256, 1024])
    ID_6 = identity_block(PB_3, [256, 256, 1024])
    ID_7 = identity_block(ID_6, [256, 256, 1024])
    ID_8 = identity_block(ID_7, [256, 256, 1024])
    ID_9 = identity_block(ID_8, [256, 256, 1024])
    ID_10 = identity_block(ID_9, [256, 256, 1024])
    PB_4 = projection_block(ID_10, [512, 512, 2048])
    ID_11 = identity_block(PB_4, [512, 512, 2048])
    ID_12 = identity_block(ID_11, [512, 512, 2048])
    pool_2 = K.layers.AveragePooling2D(pool_size=7,
                                       strides=1)(ID_12)
    output = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=w)(pool_2)
    return K.Model(inputs=inputs, outputs=output)
