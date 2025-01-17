#!/usr/bin/env python3
"""
Preprocessing data tools for Bitcoin price forecast
"""


import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def dataset_split(dataset, target, start_index, end_index, history_size,
                  target_size, step, single_step=False):
    """
    Function to get the time window
    Returns: datasets with the 24 hour window for model training
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        # Reshape data from (history_size,) to (history_size, 1)
        if (dataset.shape[1] == 1):
            data.append(np.reshape(dataset[indices], (history_size, 1)))
        else:
            data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def data_preprocessing(path_file):
    """
    Preprocess the data from the dataset.
            - path_file: where is the csv is
    Returns:
            - dataset: np.ndarray of shape (, 6) represents the number of data
                per feature of the table.
            - features: containing the features of the datframe
            - x_train: training dataset
            - y_train: labels for the training process
            - x_val: datapoints of the validation dataset
            - y_val: labels for the validation process
    """
    df = pd.read_csv(path_file)
    # removing the NaN values, invalid values
    df = df.dropna()
    # getting the decoded date
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], infer_datetime_format=True, unit='s')
    # convert the time serie per each hour
    df = df.set_index('Timestamp')
    # selecting the range from the dataset of the data to work with
    df = df.resample("H").agg({"Open": "mean", "High": "mean",
                               "Low": "mean", "Close": "mean",
                               "Volume_(BTC)": "sum",
                               "Volume_(Currency)": "sum",
                               "Weighted_Price": "mean"})
    df = df.dropna()
    df = df.iloc[-17000:]
    # selecting the features to take into account
    features_consider = ['Low', 'High',
                         'Volume_(Currency)',
                         'Weighted_Price']
    features = df[features_consider]
    # print(features.index)
    # features.index = lastYear_data['Timestamp']
    TRAIN_SPLIT = 10008
    # tf.random.set_seed(13)
    # standardize the dataset using the mean and standard deviation
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset - data_mean) / data_std

    past_history = 24
    future_target = 0
    STEP = 1
    x_train, y_train = dataset_split(dataset, dataset[:, 3],
                                     0, TRAIN_SPLIT,
                                     past_history,
                                     future_target, STEP,
                                     single_step=True)
    x_val, y_val = dataset_split(dataset, dataset[:, 3],
                                 TRAIN_SPLIT, None,
                                 past_history,
                                 future_target, STEP,
                                 single_step=True)

    return dataset, features, x_train, y_train, x_val, y_val
