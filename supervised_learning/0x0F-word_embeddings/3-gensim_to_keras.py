#!/usr/bin/env python3
"""
Function that converts gensim word2vec model to Keras Embedding layer
"""


def gensim_to_keras(model):
    """
    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
