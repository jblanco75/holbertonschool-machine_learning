#!/usr/bin/env python3
"""
Function that creates a variational autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Variational autoencoder attempt
    """
    inputs = keras.Input(shape=(input_dims,))
    h = inputs
    for node in range(1, len(hidden_layers)):
        h = keras.layers.Dense(hidden_layers[node], activation='relu')(inputs)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(shape=(z_mean[0], latent_dims),
                                              mean=0.0, stddev=1.0)
        return z_mean + keras.exp(z_log_sigma) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dims,), name='z_sampling')
    for node in range(len(hidden_layers) - 1, -1, -1):
        x = keras.layers.Dense(hidden_layers[node],
                               activation='relu')(latent_inputs)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, outputs)

    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_sigma - keras.square(z_mean) - keras.exp(z_log_sigma)
    kl_loss = keras.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
