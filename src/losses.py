from abc import abstractmethod

import numpy as np


class Loss:
    @staticmethod
    @abstractmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (np.array, np.array):
        pass


class MSELoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return 1 / 2 * np.mean(e ** 2)

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return -1 / len(e) * (x.T @ e)

    @staticmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (np.array, np.array):
        e = y - x @ w
        return 1 / 2 * np.mean(e ** 2), -1 / len(e) * (x.T @ e)


class RidgeLoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        regularizer_loss = kwargs['lambda_'] * np.dot(w.T, w)
        return 1 / 2 * np.mean(e ** 2) + regularizer_loss

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        regularizer_grad = 2 * kwargs['lambda_'] * w
        return -1 / len(e) * (x.T @ e) + regularizer_grad

    @staticmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (np.array, np.array):
        e = y - x @ w
        loss = 1 / 2 * np.mean(e ** 2) + kwargs['lambda_'] * np.dot(w.T, w)
        grad = -1 / len(e) * (x.T @ e) + 2 * kwargs['lambda_'] * w
        return loss, grad


class MAELoss(Loss):
    @staticmethod
    def loss(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return np.mean(np.abs(e))

    @staticmethod
    def grad(x: np.array, y: np.array, w: np.array, **kwargs) -> np.array:
        e = y - x @ w
        return -1 / len(e) * (x.T @ np.sign(e))

    @staticmethod
    def eval(x: np.array, y: np.array, w: np.array, **kwargs) -> (np.array, np.array):
        e = y - x @ w
        return np.mean(np.abs(e)), -1 / len(e) * (x.T @ np.sign(e))
