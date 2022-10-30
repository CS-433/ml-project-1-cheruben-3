# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


def load_data(file_path: str) -> (np.array, np.array, np.array, np.array):
    """Loads the dataset from a specified path

    Args:
        file_path: string, the path to the dataset (csv-file)

    Returns:
        4-tuple of the following:
        - A numpy array of IDs corresponding to each sample
        - A numpy array containing the labels of the samples (Filled with '?' in case of the test set)
        - A numpy array containing each feature's name
        - A numpy array containing the features corresponding to each sample

    """
    feature_names = np.genfromtxt(file_path, delimiter=",", dtype=str, max_rows=1)[2:]
    labels = np.genfromtxt(
        file_path, delimiter=",", usecols=[1], skip_header=1, dtype=str
    )
    ids = np.genfromtxt(file_path, delimiter=",", usecols=[0], skip_header=1, dtype=int)
    raw_features = np.genfromtxt(file_path, delimiter=",", skip_header=1)[:, 2:]
    return ids, labels, feature_names, raw_features


def train_test_split(X, y, train_proportion):
    """Decomposes (X,y) into (X_train, y_train) and (X_test, y_test) with a certain proportion

    Args:
        X: The features of the samples
        y: The labels of the samples
        train_proportion: Approximate proportion of datapoints to have in the training dataset v.s. the testing dataset

    Returns:
        4-tuple of the following:
        - A numpy array representing the features of the samples used for training
        - A numpy array representing the features of the samples used for testing
        - A numpy array representing the labels of the samples used for training
        - A numpy array representing the labels of the samples used for testing
    """
    assert X.shape[0] == y.shape[0]
    indices = np.random.permutation(X.shape[0])
    cutoff_idx = int(indices.shape[0] * train_proportion)
    train_idx, test_idx = indices[:cutoff_idx], indices[cutoff_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Given in lab 4
def build_k_indices(X, y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        X:      shape=(250000, 33)
        y:      shape=(250000, 1)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    assert X.shape[0] == y.shape[0]
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# Cross-validation method
def cross_validation_method(X, y, k_indices, k, method, param):
    """return the loss of methods for a fold corresponding to k_indices
    
    Args:
        X:          shape=(250000, 33)
        y:          shape=(250000, 1)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        method:     function we are going to test
        param:      dictionary parameters for such method

    Returns:
        train losses  

    """

    # First we divide the data between the train and test set obtained in k_indices
    ### Test_idx is k_indices[k] and train_idx is the rest of the indexes in a list 
    test_idx = k_indices[k]
    train_idx = k_indices[np.arange(len(k_indices)) != k].reshape(-1)
    ### We proceed with the division
    X_train, X_test, Y_train, Y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    losses = []
    # After this, we test the method

    if (method.__name__ == 'ridge_regression'):
        w, loss = method(Y_train, X_train, lambda_=param["lambda_"])
        losses.append(loss)

    elif (method.__name__ == 'reg_logistic_regression'):
        w, loss = method(Y_train, X_train, lambda_=param["lambda_"], initial_w=np.zeros(shape=(X_train.shape[1], 1)), max_iters=param["max_iters"], gamma=param["gamma"])
        losses.append(loss)

    elif (method.__name__ == 'reg_logistic_regression_AGDR'):
        w, loss = method(Y_train, X_train, lambda_=param["lambda_"], initial_w=np.zeros(shape=(X_train.shape[1], 1)), max_iters=param["max_iters"], gamma=param["gamma"])
        losses.append(loss)

    else:
        print("No such name")


    return losses


# k fold division function
def k_fold_division(X, y, k_fold, seed):
    """return all subsets of training and test sets

    Args:
        X:          shape=(250000, 33)
        y:          shape=(250000, 1)
        k-fold:     int with the number of k_folds used in build_k_indices()
        seed:       int for the seed of randomize used in build_k_indices()

    Returns:
        Subsets of training and test sets

    """

    # First we get the k_indices:  2D array returned by build_k_indices()
    k_indices = build_k_indices(X, y, k_fold, seed)

    X_train_sets = []
    X_test_sets = []
    Y_train_sets = []
    Y_test_sets = []
    # We get all the sets of the k_folds
    for k in range(k_fold):
        # Division of the data into the train and test set obtained in k_indices with build_k_indices
        ### Test_idx is k_indices[k] and train_idx is the rest of the indexes in a list
        test_idx = k_indices[k]
        train_idx = k_indices[np.arange(len(k_indices)) != k].reshape(-1)
        ### We proceed with the division
        X_train_sets.append(X[train_idx])
        X_test_sets.append(X[test_idx])
        Y_train_sets.append(y[train_idx])
        Y_test_sets.append(y[test_idx])

    return X_train_sets, X_test_sets, Y_test_sets, Y_test_sets


def standardize(
        x: np.array,
        columns: list[int] = None,
        column_means: np.array = None,
        column_stds: np.array = None,
) -> (np.array, np.array):
    """In-place standardizing of the specified columns of a data matrix x

    Args:
        x: 2d numpy-array where rows represent samples and columns represent features
        columns: the specified column indices to standardize
        column_means: if not empty, use these corresponding column means for standardization
        (e.g. for standardizing test dataset according to train dataset distribution)
        column_stds:if not empty, use these corresponding column standard deviations for standardization

    Returns:
        -> Does in-place standardization
        2-tuple of the following:
        - The mean of each column before standardizing
        - The standard deviation of each column before standardizing

    """
    if columns is None:
        columns = np.arange(x.shape[1])

    specified_x = x[:, columns]

    if column_means is None:
        column_means = np.nanmean(specified_x, axis=0)
    if column_stds is None:
        column_stds = np.nanstd(specified_x, axis=0)

    x[:, columns] = (specified_x - column_means) / column_stds
    return column_means, column_stds


def get_interaction_terms_columns(X: np.array, co_linear_threshold: float):
    """ Gets list of pairs of features that have an absolute correlation higher than a certain threshold

    Args:
        X: 2d numpy-array where rows represent samples and columns represent features
        co_linear_threshold: the minimum correlation for a pair of features to be added to the return list

    Returns:
        List of tuples, where each element in a tuple represents the index of a feature, and the tuple represents
        collinearity

    """
    corr_matrix = np.corrcoef(X, rowvar=False)
    co_linear_feature_columns = []
    for i in range(len(corr_matrix)):
        for j in range(i, len(corr_matrix)):
            if np.abs(corr_matrix[i][j]) >= co_linear_threshold:
                co_linear_feature_columns.append((i, j))

    return co_linear_feature_columns


def add_interaction_terms_columns(X: np.array, co_linear_feature_columns: list):
    """ Creates interaction term features of feature pairs that are considered collinear

    Args:
        X: 2d numpy-array where rows represent samples and columns represent features
        co_linear_feature_columns: List of tuples, where each element in a tuple represents the index of a feature,
        and the tuple represents collinearity

    Returns:
        2d numpy-array where rows represent samples and columns represent features. The original data X is copied into
        this array, but the array is extended by additional features from the interaction terms
    """
    X_co_linear_features = np.zeros(
        (X.shape[0], X.shape[1] + len(co_linear_feature_columns)))
    X_co_linear_features[:, :X.shape[1]] = X
    start_idx = X.shape[1]
    for i, (f1_idx, f2_idx) in enumerate(co_linear_feature_columns):
        X_co_linear_features[:, start_idx + i] = X[:, f1_idx] * X[:, f2_idx]

    return X_co_linear_features


def one_hot_encode(
        x: np.array, columns: list[int], feature_names: np.array
) -> (np.array, np.array):
    """One-hot encodes certain categorical features

    Args:
        x: 2d numpy-array where rows represent samples and columns represent features
        columns: The columns that need to be one-hot encoded
        feature_names: The names of the features that need to be ont-hot encoded (used to generate new column names)

    Returns:
        2d numpy-array where rows represent samples and columns represent features. The array remains identical to x,
        but with features representing the one-hot encodings of columns replacing the column feature
    """
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


def build_poly(x, degree):
    """Builds polynomial features"""
    res = x.copy()
    for p in range(2, degree + 1):
        res = np.concatenate((res, np.power(x, p)), axis=1)
    return res
