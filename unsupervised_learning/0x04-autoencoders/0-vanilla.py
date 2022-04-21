#!/usr/bin/env python3
"""
Function that creates an autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
      the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
      encoder is the encoder model
      decoder is the decoder model
      auto is the full autoencoder model
    """
    input_layer = keras.Input(input_dims)
    layer = input_layer
    for node in range(len(hidden_layers)):
        layer = keras.layers.Dense(hidden_layers[node],
                                   activation='relu')(layer)
    latent_layer = keras.layers.Dense(latent_dims, activation='relu')(layer)
    encoder = keras.Model(input_layer, latent_layer)

    latent_input = keras.Input(latent_dims)
    layer = latent_input
    for node in range(len(hidden_layers) - 1, -1, -1):
        layer = keras.layers.Dense(hidden_layers[node],
                                   activation='relu')(layer)
    last_layer = keras.layers.Dense(input_dims,
                                    activation='sigmoid')(layer)
    decoder = keras.Model(latent_layer, last_layer)
    auto = keras.Model(input_layer, decoder(encoder(input_layer)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
