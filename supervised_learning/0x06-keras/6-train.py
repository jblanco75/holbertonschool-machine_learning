#!/usr/bin/env python3
"""
Function that trains a model using mini-batch gradient descent
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains the model using early stopping
    """
    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience,
                                             mode='min')
    else:
        callback = None
    return network.fit(x=data, y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=verbose,
                       callbacks=callback,
                       shuffle=shuffle)
