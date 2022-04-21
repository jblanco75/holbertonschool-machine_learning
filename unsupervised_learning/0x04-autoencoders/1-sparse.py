#!/usr/bin/env python3
"""
Function that creates a sparse autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
      the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent
    space representation
    lambtha is the regularization parameter used for L1 regularization
    on the encoded output
    Returns: encoder, decoder, auto
      encoder is the encoder model
      decoder is the decoder model
      auto is the sparse autoencoder model
    """
    input_layer = keras.Input(shape=(input_dims, ))
    regularizer = keras.regularizers.L1(lambtha)
    encode_layer = keras.layers.Dense(hidden_layers[0],
                                      activity_regularizer=regularizer,
                                      activation='relu')(input_layer)
    for enc in range(1, len(hidden_layers)):
        encode_layer = keras.layers.Dense(hidden_layers[0],
                                          activity_regularizer=regularizer,
                                          activation='relu')(encode_layer)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encode_layer)
    encoder = keras.Model(inputs=input_layer, outputs=latent)

    input_decoder = keras.Input(shape=(latent_dims, ))
    decode_layer = keras.layers.Dense(hidden_layers[-1],
                                      activation='relu')(input_decoder)
    for dec in range(len(hidden_layers) - 2, -1, -1):
        decode_layer = keras.layers.Dense(hidden_layers[dec],
                                          activation='relu')(decode_layer)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decode_layer)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_out = encoder(input_layer)
    decoder_out = decoder(encoder_out)
    auto = keras.Model(inputs=input_layer, outputs=decoder_out)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
