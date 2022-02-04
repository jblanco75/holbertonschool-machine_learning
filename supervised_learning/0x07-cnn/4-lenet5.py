#!/usr/bin/env python3
"""
Function that builds a modified version of the
LeNet-5 architecture using tensorflow 1
"""


import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    x is a tf.placeholder of shape (m, 28, 28, 1) containing
    the input images for the network
      m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the
    one-hot labels for the network
    The model should consist of the following layers in order:
      Convolutional layer with 6 kernels of shape 5x5 with same padding
      Max pooling layer with kernels of shape 2x2 with 2x2 strides
      Convolutional layer with 16 kernels of shape 5x5 with valid padding
      Max pooling layer with kernels of shape 2x2 with 2x2 strides
      Fully connected layer with 120 nodes
      Fully connected layer with 84 nodes
      Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their
    kernels with the he_normal initialization
    method: tf.keras.initializers.VarianceScaling(scale=2.0)
    All hidden layers requiring activation
    should use the relu activation function
      you may import tensorflow.compat.v1 as tf
      you may NOT use tf.keras only for the he_normal method.
    Returns:
      a tensor for the softmax activated output
      a training operation that utilizes Adam optimization
      (with default hyperparameters)
      a tensor for the loss of the netowrk
      a tensor for the accuracy of the network
    """
    weights_init = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv_1 = tf.layers.Conv2D(filters=6,
                              kernel_size=(5, 5),
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=weights_init)(x)
    max_pool_1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv_1)
    conv_2 = tf.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding='valid',
                              activation=tf.nn.relu,
                              kernel_initializer=weights_init)(max_pool_1)
    max_pool_2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv_2)

    flat = tf.layers.Flatten()(max_pool_2)

    fully_1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                              kernel_initializer=weights_init)(flat)
    fully_2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                              kernel_initializer=weights_init)(fully_1)
    fully_3 = tf.layers.Dense(units=10,
                              kernel_initializer=weights_init)(fully_2)

    softmax_output = tf.nn.softmax(fully_3)
    loss = tf.losses.softmax_cross_entropy(y, fully_3)
    train = tf.train.AdamOptimizer().minimize(loss)
    equality = tf.equal(tf.argmax(fully_3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return softmax_output, train, loss, accuracy
