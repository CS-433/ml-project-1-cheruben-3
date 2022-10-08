from abc import abstractmethod

import numpy as np


def confusion_matrix_statistics(y_pred: np.array, y: np.array, positive_value: int = 1, negative_value: int = 0):
    positive_idxs = y == positive_value
    negative_idxs = y == negative_value

    tp = np.sum(y_pred[positive_idxs] == positive_value).item()
    fn = np.sum(y_pred[positive_idxs] == negative_value).item()
    fp = np.sum(y_pred[negative_idxs] == positive_value).item()
    tn = np.sum(y_pred[negative_idxs] == negative_value).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)  # The number of true positives over all positives
    precision = tp / (tp + fp)  # The number of true positives over all positive predictions
    f1_score = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)  # The number of false positives over all negatives

    return tp, fp, fn, tn, accuracy, recall, precision, f1_score, fpr


class Loss:
    @staticmethod
    @abstractmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> float:
        pass

    @staticmethod
    @abstractmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (float, np.array):
        pass


class MSELoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> float:
        e = y - x @ w
        loss = 1 / 2 * np.mean(e ** 2)
        return loss.item()

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return -1 / len(e) * (x.T @ e)

    @staticmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (float, np.array):
        e = y - x @ w
        loss = 1 / 2 * np.mean(e ** 2)
        return loss.item(), -1 / len(e) * (x.T @ e)


class RidgeLoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> float:
        e = y - x @ w
        regularizer_loss = kwargs['lambda_'] * np.dot(w.T, w)
        total_loss = 1 / 2 * np.mean(e ** 2) + regularizer_loss
        return total_loss.item()

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        regularizer_grad = 2 * kwargs['lambda_'] * w
        return -1 / len(e) * (x.T @ e) + regularizer_grad

    @staticmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (float, np.array):
        e = y - x @ w
        loss = 1 / 2 * np.mean(e ** 2) + kwargs['lambda_'] * np.dot(w.T, w)
        grad = -1 / len(e) * (x.T @ e) + 2 * kwargs['lambda_'] * w
        return loss.item(), grad


class MAELoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> float:
        e = y - x @ w
        loss = np.mean(np.abs(e))
        return loss.item()

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return -1 / len(e) * (x.T @ np.sign(e))

    @staticmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (float, np.array):
        e = y - x @ w
        loss = np.mean(np.abs(e))
        return loss.item(), -1 / len(e) * (x.T @ np.sign(e))
