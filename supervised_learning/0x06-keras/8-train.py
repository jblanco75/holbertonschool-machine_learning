#!/usr/bin/env python3
"""
Function that trains a model using mini-batch gradient descent
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                save_best=False, filepath=None,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Also saves the best iteration of the model
    """
    def scheduler(alpha, decay_rate):
        return K.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=alpha,
            decay_rate=decay_rate)
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   mode='min'))
    elif learning_rate_decay and validation_data:
        callbacks.append(K.callbacks.LearningRateScheduler(schedule=scheduler,
                                                           verbose=1))
    elif save_best and validation_data:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                     save_best_only=save_best))
    else:
        callbacks = None
    return network.fit(x=data, y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=verbose,
                       callbacks=callbacks,
                       shuffle=shuffle)
