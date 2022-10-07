import numpy as np


def least_squares_GD(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float) -> [np.array, float]:
    pass


def least_squares_SGD(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float) -> [np.array, float]:
    pass


def least_squares(y: np.array, tx: np.array) -> [np.array, float]:
    pass


def ridge_regression(y: np.array, tx: np.array, lambda_: float) -> [np.array, float]:
    pass


def logistic_regression(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float) -> \
        [np.array, float]:
    pass


def reg_logistic_regression(y: np.array, tx: np.array, lambda_: float, initial_w: np.array, max_iters: int, gamma: float
                            ) -> [np.array, float]:
    pass
