# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(file_path):
    """Load the raw data"""
    feature_names = np.genfromtxt(file_path, delimiter=',', dtype=str, max_rows=1)[2:]
    labels = np.genfromtxt(file_path, delimiter=',', usecols=[1], skip_header=1, dtype=str)
    ids = np.genfromtxt(file_path, delimiter=',', usecols=[0], skip_header=1, dtype=int)
    raw_features = np.genfromtxt(file_path, delimiter=',', skip_header=1)[:, 2:]
    return ids, labels, feature_names, raw_features


def standardize(x):
    """Standardize the columns of a data matrix x"""
    column_means = np.mean(x, axis=0)
    column_stds = np.std(x, axis=0)
    return (x - column_means) / column_stds, column_means, column_stds

