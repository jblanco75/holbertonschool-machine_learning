#!/usr/bin/env python3
"""
Function that trains a model using mini-batch gradient descent
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Also trains the model with learning rate decay
    """
    def scheduler(epoch):
        return alpha / (1 + decay_rate * epoch)
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   mode='min'))
    elif learning_rate_decay and validation_data:
        callbacks.append(K.callbacks.LearningRateScheduler(schedule=scheduler,
                                                           verbose=1))
    else:
        callbacks = None
    return network.fit(x=data, y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=verbose,
                       callbacks=callbacks,
                       shuffle=shuffle)
