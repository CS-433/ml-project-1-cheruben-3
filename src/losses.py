from abc import abstractmethod

import numpy as np


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
