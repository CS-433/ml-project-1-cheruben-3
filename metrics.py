from abc import abstractmethod

import numpy as np


def confusion_matrix_statistics(
    y_pred: np.array, y: np.array, positive_value: int = 1, negative_value: int = 0
):
    """Returns statistics derived from a confusion matrix of a binary classification problem

    Args:
        y_pred: numpy array containing the estimated labels of an algorithm
        y: numpy array containing the actual labels
        positive_value: the value encoding a positive sample
        negative_value: the value encoding a negative sample

    Returns:
        9-tuple containing the following:
        - The number of true-positive samples
        - The number of false-positive samples
        - The number of false-negative samples
        - The number of true-negative samples
        - The accuracy
        - The recall
        - The precision
        - The f1-score
        - The false-positive rate

    """
    positive_idxs = y == positive_value
    negative_idxs = y == negative_value

    # Computes confusion matrix
    tp = np.sum(y_pred[positive_idxs] == positive_value).item()
    fn = np.sum(y_pred[positive_idxs] == negative_value).item()
    fp = np.sum(y_pred[negative_idxs] == positive_value).item()
    tn = np.sum(y_pred[negative_idxs] == negative_value).item()

    # Derives some statistics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)  # The number of true positives over all positives
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (
            tp + fp
        )  # The number of true positives over all positive predictions
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)  # The number of false positives over all negatives

    return tp, fp, fn, tn, accuracy, recall, precision, f1_score, fpr


class Loss:
    """
    Abstract class creating a representation of a loss function
    """

    @staticmethod
    @abstractmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the loss of some loss function

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: (Optional) any additional information to be included in the computation (e.g. regularization
            constant)

        Returns:
            The loss of a certain model on (x, y)
        """
        pass

    @staticmethod
    @abstractmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the gradient of some loss function with respect to the model's weights

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: (Optional) any additional information to be included in the computation (e.g. regularization
            constant)

        Returns:
            The gradient of the loss function with respect to the weights of a certain model on (x, y)
        """
        pass


class MSELoss(Loss):
    """
    Implementation of the MSE loss function
    """

    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the loss of MSE

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The MSE loss of a certain model on (x, y)
        """
        e = y - x @ w
        loss = 1 / 2 * np.mean(e**2)
        return np.array(loss.item())

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the gradient of the MSE loss with respect to the model's weights

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The gradient of the MSE loss function with respect to the weights of a certain model on (x, y)
        """
        e = y - x @ w
        return -1 / len(e) * (x.T @ e)


class RidgeLoss(Loss):
    """
    Implementation of the ridge regression's loss function
    """

    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the loss of the ridge regression loss

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The MSE loss of a certain model on (x, y)
        """
        e = y - x @ w
        total_loss = 1 / 2 * np.mean(e**2)
        return np.array(total_loss.item())

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the gradient of ridge regression's loss with respect to the model's weights

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: {'lambda_': FLOAT} to represent the regularization parameter

        Returns:
            The gradient of ridge regression's loss function with respect to the weights of a certain model on (x, y)
        """
        e = y - x @ w
        regularizer_grad = 2 * kwargs["lambda_"] * w
        return -1 / len(e) * (x.T @ e) + regularizer_grad


class MAELoss(Loss):
    """
    Implementation of the MAE loss function
    """

    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the loss of the MAE loss

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The MAE loss of a certain model on (x, y)
        """
        e = y - x @ w
        loss = np.mean(np.abs(e))
        return np.array(loss.item())

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the gradient of the MAE loss with respect to the model's weights

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The gradient of the MAE loss function with respect to the weights of a certain model on (x, y)
        """
        e = y - x @ w
        return -1 / len(e) * (x.T @ np.sign(e))


class RegLogisticRegressionLoss(Loss):
    """
    Implementation of the regularized logistic regression's loss function
    """

    # For numerical stability of the logarithm
    EPS = 1e-8

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        """Evaluated the sigmoid function at a certain point

        Args:
            x: A numpy array of points to run through the sigmoid function

        Returns:
            A numpy array of results of the sigmoid function for all datapoints x
        """
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the loss for regularized logistic regression

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The regularized logistic loss of a certain model on (x, y)
        """
        return LogisticRegressionLoss.loss(x, y, w)

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the gradient of the regularized logistic loss with respect to the model's
        weights

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: {'lambda_': FLOAT} to represent the regularization parameter

        Returns:
            The gradient of the regularized logistic loss function with respect to the weights of a certain model on
            (x, y)
        """
        y_pred = RegLogisticRegressionLoss.sigmoid(x @ w)
        regularizer_grad = 2 * kwargs["lambda_"] * w
        return 1 / len(y) * x.T @ (y_pred - y) + regularizer_grad


class LogisticRegressionLoss(Loss):
    """
    Implementation of the logistic regression's loss function
    """

    # For numerical stability of the logarithm
    EPS = 1e-8

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        """Evaluated the sigmoid function at a certain point

        Args:
            x: A numpy array of points to run through the sigmoid function

        Returns:
            A numpy array of results of the sigmoid function for all datapoints x
        """
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the loss for logistic regression

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The logistic loss of a certain model on (x, y)
        """
        y_pred = LogisticRegressionLoss.sigmoid(x @ w)
        total_loss = (
            -1
            / len(y)
            * (
                y.T @ np.log(y_pred + LogisticRegressionLoss.EPS)
                + (1 - y.T) @ np.log(1 - y_pred + LogisticRegressionLoss.EPS)
            )
        )
        return np.array(total_loss.item())

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        """Defines a formula to calculate the gradient of the logistic loss with respect to the model's weights

        Args:
            x: The features of the samples
            y: The labels of the samples
            w: The weights of the algorithm
            **kwargs: ignored

        Returns:
            The gradient of the logistic loss function with respect to the weights of a certain model on (x, y)
        """
        y_pred = LogisticRegressionLoss.sigmoid(x @ w)
        return 1 / len(y) * x.T @ (y_pred - y)
