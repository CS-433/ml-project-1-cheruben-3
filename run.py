import csv
import os

import numpy as np

from helpers import (
    load_data,
    one_hot_encode,
    standardize,
    get_interaction_terms_columns,
    add_interaction_terms_columns,
)
from implementations import reg_logistic_regression_AGDR
from metrics import LogisticRegressionLoss

# PLEASE RUN FROM INSIDE THE "SRC" DIRECTORY

if __name__ == "__main__":

    data_directory = "../data"
    train_dataset_path = os.path.join(data_directory, "data/train.csv")
    public_test_dataset_path = os.path.join(data_directory, "data/test.csv")

    # Loading the data
    _, Y_train_public, feature_names, X_train_public = load_data(train_dataset_path)
    ids_test_public, _, _, X_test_public = load_data(public_test_dataset_path)

    # We need to deal with -999 somehow (missing values)
    # First of all, we make sure they do not contribute to computing the mean and std
    # Since they are floats, we add an epsilon against numerical errors
    EPSILON = 1e-4
    mask_train = np.abs(X_train_public + 999) <= EPSILON
    mask_test = np.abs(X_test_public + 999) <= EPSILON

    X_train_public[mask_train] = np.nan
    X_test_public[mask_test] = np.nan

    print(
        "Proportion of missing values:",
        np.sum(mask_train) / (mask_train.shape[0] * mask_train.shape[1]),
    )
    print(
        "Proportion of missing values:",
        np.sum(mask_test) / (mask_test.shape[0] * mask_test.shape[1]),
    )

    # We will standardize the data based on the mean and standard deviation of the public train dataset
    # It ignores NaN values for computing the mean and std for the standardization
    # ! The method standardizes in-place !
    continuous_column_idxs = np.where(feature_names != "PRI_jet_num")[0]
    column_means, column_stds = standardize(X_train_public, continuous_column_idxs)
    _, _ = standardize(X_test_public, continuous_column_idxs, column_means, column_stds)

    # Finally, we set the NaNs to the mean of the standardized dataset, namely, 0
    X_train_public = np.nan_to_num(X_train_public, nan=0.0)
    X_test_public = np.nan_to_num(X_test_public, nan=0.0)

    # Then we will need to notice the discrete-valued column, since this needs to be one-hot encoded
    # In our dataset, only "PRI_jet_num" is discrete.
    discrete_column_idxs = np.where(feature_names == "PRI_jet_num")[0]

    # Update the features by one-hot encoding the discrete ones, but only update the feature names at the end
    # They will be the same for the train and test set anyway
    X_train_public, _ = one_hot_encode(
        X_train_public, discrete_column_idxs, feature_names
    )
    X_test_public, feature_names = one_hot_encode(
        X_test_public, discrete_column_idxs, feature_names
    )

    # Since this is a binary classification problem, we do not need to one-hot encode the y-vector, but we can just use
    # binary values
    positive_sample = "s"
    negative_sample = "b"
    Y_train_public = np.expand_dims(
        (Y_train_public == positive_sample).astype(np.int32), axis=1
    )

    # ! There are quite some more positive than negative samples, maybe we could try to weigh negative samples more or
    # something? !
    print("\nNumber of positive samples:", np.sum(Y_train_public))
    print("Number of negative samples:", len(Y_train_public) - np.sum(Y_train_public))

    # The best performing model's features
    co_linear_feature_columns = get_interaction_terms_columns(
        X_train_public, co_linear_threshold=0.05
    )
    print(
        "\nWe have found",
        len(co_linear_feature_columns),
        "interaction terms with at least that correlation",
    )
    X_train_public = add_interaction_terms_columns(
        X_train_public, co_linear_feature_columns
    )
    X_test_public = add_interaction_terms_columns(
        X_test_public, co_linear_feature_columns
    )

    # Training model
    print("\nTraining model...")
    print(
        "Note: this might take a bit, since we are training on a (250000, 250+)-sized dataset"
    )
    max_iterations = 1000
    gamma = 0.1
    w, loss = reg_logistic_regression_AGDR(
        Y_train_public,
        X_train_public,
        lambda_=0,
        initial_w=np.zeros(shape=(X_train_public.shape[1], 1)),
        max_iters=max_iterations,
        gamma=gamma,
    )
    print("Loss:", loss)
    print("Training complete!\n")

    # Generating submission
    print("Generating submission file...")
    best_cutoff = 0.5
    predictions = (
        LogisticRegressionLoss.sigmoid(X_test_public @ w) > best_cutoff
    ).astype(np.int32)

    submission_file_name = "data/best_submission.csv"
    with open(
        os.path.join(data_directory, submission_file_name),
        mode="w",
        newline="",
        encoding="utf-8",
    ) as submission_file:
        writer = csv.writer(submission_file, delimiter=",")
        writer.writerow(["Id", "Prediction"])
        for i, prediction in enumerate(predictions):
            writer.writerow([ids_test_public[i], 1 if prediction else -1])
    print("Submission has been generated terminating...")
