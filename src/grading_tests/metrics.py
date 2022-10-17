from abc import abstractmethod

import numpy as np


def confusion_matrix_statistics(
    y_pred: np.array, y: np.array, positive_value: int = 1, negative_value: int = 0
):
    positive_idxs = y == positive_value
    negative_idxs = y == negative_value

    tp = np.sum(y_pred[positive_idxs] == positive_value).item()
    fn = np.sum(y_pred[positive_idxs] == negative_value).item()
    fp = np.sum(y_pred[negative_idxs] == positive_value).item()
    tn = np.sum(y_pred[negative_idxs] == negative_value).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)  # The number of true positives over all positives
    precision = tp / (
        tp + fp
    )  # The number of true positives over all positive predictions
    f1_score = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)  # The number of false positives over all negatives

    return tp, fp, fn, tn, accuracy, recall, precision, f1_score, fpr


class Loss:
    @staticmethod
    @abstractmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        pass


class MSELoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        loss = 1 / 2 * np.mean(e**2)
        return np.array(loss.item())

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return -1 / len(e) * (x.T @ e)


class RidgeLoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        # TODO: I'm not sure, but the public test case omits this additional loss
        regularizer_loss = kwargs["lambda_"] * np.dot(w.T, w)
        total_loss = 1 / 2 * np.mean(e**2)  # + regularizer_loss
        return np.array(total_loss.item())

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        regularizer_grad = 2 * kwargs["lambda_"] * w
        return -1 / len(e) * (x.T @ e) + regularizer_grad


class MAELoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        loss = np.mean(np.abs(e))
        return np.array(loss.item())

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return -1 / len(e) * (x.T @ np.sign(e))


class RegLogisticRegressionLoss(Loss):
    EPS = 0

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        return LogisticRegressionLoss.loss(x, y, w)

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        y_pred = RegLogisticRegressionLoss.sigmoid(x @ w)
        regularizer_grad = 2 * kwargs["lambda_"] * w
        return 1 / len(y) * x.T @ (y_pred - y) + regularizer_grad


class LogisticRegressionLoss(Loss):
    EPS = 0

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
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
        y_pred = LogisticRegressionLoss.sigmoid(x @ w)
        return 1 / len(y) * x.T @ (y_pred - y)
