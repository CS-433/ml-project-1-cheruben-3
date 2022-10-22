# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(file_path: str) -> (np.array, np.array, np.array, np.array):
    """Load the raw data"""
    feature_names = np.genfromtxt(file_path, delimiter=",", dtype=str, max_rows=1)[2:]
    labels = np.genfromtxt(
        file_path, delimiter=",", usecols=[1], skip_header=1, dtype=str
    )
    ids = np.genfromtxt(file_path, delimiter=",", usecols=[0], skip_header=1, dtype=int)
    raw_features = np.genfromtxt(file_path, delimiter=",", skip_header=1)[:, 2:]
    return ids, labels, feature_names, raw_features


def standardize(
    x: np.array,
    columns: list[int] = None,
    column_means: np.array = None,
    column_stds: np.array = None
) -> (np.array, np.array):
    """In-place standardizing of the specified columns of a data matrix x"""
    if columns is None:
        columns = np.arange(x.shape[1])

    specified_x = x[:, columns]

    if column_means is None:
        column_means = np.nanmean(specified_x, axis=0)
    if column_stds is None:
        column_stds = np.nanstd(specified_x, axis=0)

    x[:, columns] = (specified_x - column_means) / column_stds
    return column_means, column_stds


def one_hot_encode(
    x: np.array, columns: list[int], feature_names: np.array
) -> (np.array, np.array):
    # WARNING: Will NOT throw an error if the values are floats, it simply converts them
    # Make sure you are encoding the right columns!

    # Selects the values of the discrete columns and converts them to integers
    # Also selects the column names (since we have to make new column names)
    one_hot_cols = x[:, columns].T.astype(np.int32)
    one_hot_col_names = feature_names[columns]

    # Delete the old columns from the data (we have them stored in earlier variables)
    x = np.delete(x, columns, axis=1)
    feature_names = np.delete(feature_names, columns, axis=0)

    # Make temporary variables that we add the one-hot encodings to
    new_feature_names = []
    new_features = None

    # Compute the maximum values for each column
    maximum_value_per_col = one_hot_cols.max(axis=1)

    for i, one_hot_col in enumerate(one_hot_cols):
        max_col_val = maximum_value_per_col[i]
        for j in range(max_col_val + 1):
            # Add a new feature name for each possible value of the current column
            new_feature_names.append(one_hot_col_names[i] + "_" + str(j))
        # Create the matrix of one-hot encodings for this feature
        one_hot_matrix = np.zeros((len(one_hot_col), max_col_val + 1))
        one_hot_matrix[np.arange(len(one_hot_col)), one_hot_col] = 1

        # Concatenate the one-hot encoded matrix with the ones from previous columns
        if new_features is not None:
            new_features = np.concatenate([new_features, one_hot_matrix], axis=1)
        else:
            new_features = one_hot_matrix

    # Return the concatenated old features with the new one-hot encoded features, do the same for the column names
    return np.concatenate([x, new_features], axis=1), np.concatenate(
        [feature_names, new_feature_names], axis=0
    )


def batch_iter(x, y, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = x[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = x
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
