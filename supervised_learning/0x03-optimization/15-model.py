#!/usr/bin/env python3
"""
Script for function that builds, trains,
and saves a neural network model in tensorflow
using Adam optimization, mini-batch gradient
descent, learning rate decay, and batch normalization
"""


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """Function that returns a tensor output"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=w)
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """Function that returns an activated tensor output"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=w, name="layer")
    y = layer(prev)
    mean, variance = tf.nn.moments(y, [0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    y_norm = tf.nn.batch_normalization(y, mean, variance, offset=beta,
                                       scale=gamma, variance_epsilon=1e-8)
    return (activation(y_norm))


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward Propagation Function"""
    A = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            A = create_batch_norm_layer(A, layer_sizes[i], activations[i])
        else:
            A = create_layer(A, layer_sizes[i], activations[i])
        return A


def calculate_accuracy(y, y_pred):
    """Function that calculates the tensor accuracy"""
    equal = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """Function that calculates the tensor loss"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Function that returns a tensor optimized using Adam"""
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train = optimizer.minimize(loss)
    return train


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """"Function that returns the learning rate decay"""
    return tf.train.inverse_time_decay(alpha, global_step,
                                       decay_step, decay_rate,
                                       staircase=True)


def shuffle_data(X, Y):
    """Function that returns a shuffled matrix"""
    perm = X.shape[0]
    shuff_op = np.random.permutation(perm)
    return X[shuff_op], Y[shuff_op]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Returns: the path where the model was saved
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    step_increase = tf.assign(global_step, global_step + 1)
    alpha1 = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha1, beta1, beta2, epsilon)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        b_iter = Data_train[0].shape[0] // batch_size
        if b_iter % batch_size != 0:
            b_iter += 1
            fl = True
        else:
            fl = False
            for i in range(epochs + 1):
                cos_t, acc_t = sess.run([loss, accuracy],
                                        {x: Data_train[0], y: Data_train[1]})
                cos_v, acc_v = sess.run([loss, accuracy],
                                        {x: Data_valid[0], y: Data_valid[1]})
                print('After {} epochs:'.format(i))
                print('\tTraining Cost: {}'.format(cos_t))
                print('\tTraining Accuracy: {}'.format(acc_t))
                print('\tValidation Cost: {}'.format(cos_v))
                print('\tValidation Accuracy: {}'.format(acc_v))
                if i < epochs:
                    x_tr, y_tr = shuffle_data(Data_train[0], Data_train[1])
                    for j in range(b_iter):
                        start = j * batch_size
                        if j == b_iter - 1 and fl is True:
                            final = Data_train[0].shape[0]
                        else:
                            final = j * batch_size + batch_size
                        batch_x = x_tr[start:final]
                        batch_y = y_tr[start:final]
                        sess.run([train_op], {x: batch_x, y: batch_y})
                        if j != 0 and (j + 1) % 100 == 0:
                            batch_co, batch_ac = sess.run([loss, accuracy],
                                                          {x: batch_x, y:
                                                           batch_y})
                            print('\tStep {}:'.format(j + 1))
                            print('\t\tCost: {}'.format(batch_co))
                            print('\t\tAccuracy: {}'.format(batch_ac))
                sess.run(step_increase)
            return saver.save(sess, save_path)
