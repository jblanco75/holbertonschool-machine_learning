#!/usr/bin/env python3
"""
Function that trains a loaded neural
network model using mini-batch gradient descent
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Perform a minibatch gradient descent"""
    with tf.Session() as sess:
        saver_n = tf.train.import_meta_graph(load_path + '.meta')
        saver_n.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        b_iter = X_train.shape[0] // batch_size
        if b_iter % batch_size != 0:
            b_iter += 1
            fl = True
        else:
            fl = False
            for i in range(epochs + 1):
                cos_t, acc_t = sess.run([loss, accuracy], {x: X_train, y:
                                                           Y_train})
                cos_v, acc_v = sess.run([loss, accuracy], {x: X_valid, y:
                                                           Y_valid})
                print('After {} epochs:'.format(i))
                print('\tTraining Cost: {}'.format(cos_t))
                print('\tTraining Accuracy: {}'.format(acc_t))
                print('\tValidation Cost: {}'.format(cos_v))
                print('\tValidation Accuracy: {}'.format(acc_v))
                if i < epochs:
                    x_tr, y_tr = shuffle_data(X_train, Y_train)
                    for j in range(b_iter):
                        start = j * batch_size
                        if j == b_iter - 1 and fl is True:
                            final = X_train.shape[0]
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
            return saver_n.save(sess, save_path)
