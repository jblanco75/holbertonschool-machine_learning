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
    """
    Returns: the tensor output of the layer
    """
    weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=weights)
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    Returns: a tensor of the activated output for the layer
    """
    k_init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, kernel_initializer=k_init)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True,
                        name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True,
                       name='beta')
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(layer(prev), mean, variance,
                                           beta, gamma, epsilon)
    return activation(batch_norm)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Returns: the prediction of the network in tensor form
    """
    output = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for layer in range(1, len(layer_sizes)):
        if layer != len(layer_sizes) - 1:
            output = create_batch_norm_layer(output, layer_sizes[layer],
                                             activations[layer])
        else:
            output = create_layer(output, layer_sizes[layer],
                                  activations[layer])
    return output


def calculate_accuracy(y, y_pred):
    """
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """Function that calculates the tensor loss"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Returns: the Adam optimization operation
    """
    adam_optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    trainer = adam_optimizer.minimize(loss)
    return trainer


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Returns: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha, global_step,
                                       decay_step, decay_rate,
                                       staircase=True)


def shuffle_data(X, Y):
    """
    Returns: the shuffled X and Y matrices
    """
    shuffled = np.random.permutation(X.shape[0])
    return X[shuffled, :], Y[shuffled, :]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Returns: the path where the model was saved
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid
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

        m = X_train.shape[0]

        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        for epoch in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            if epoch < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)
                for batch in range(n_batches):
                    start = batch * batch_size
                    end = (batch + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]
                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    sess.run(train_op, feed_dict=next_train)

                    if (batch + 1) % 100 == 0 and batch != 0:
                        loss_mini_batch = sess.run(loss,
                                                   feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(batch + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))

            sess.run(tf.assign(global_step, global_step + 1))

        return saver.save(sess, save_path)
