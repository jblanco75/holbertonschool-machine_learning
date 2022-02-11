#!/usr/bin/env python3
"""
Function that builds the DenseNet-121 architecture as
described in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer
