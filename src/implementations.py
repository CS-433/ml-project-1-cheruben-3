import numpy as np
from tqdm import tqdm

from helpers import batch_iter
from metrics import (
    MSELoss,
    RidgeLoss,
    LogisticRegressionLoss,
    RegLogisticRegressionLoss,
)


def mean_squared_error_gd(
    y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float
) -> (np.array, float):
    """
    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    w = initial_w

    for n_iter in range(max_iters):
        grad = MSELoss.grad(tx, y, w)
        w -= gamma * grad

        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, MSELoss.loss(tx, y, w)


def mean_squared_error_sgd(
    y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float
) -> (np.array, float):
    """
    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """

    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(tx, y, batch_size=1, num_batches=1):
            grad = MSELoss.grad(tx_batch, y_batch, w)
            w = w - gamma * grad

            # print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, MSELoss.loss(tx, y, w)


def least_squares(y: np.array, tx: np.array) -> (np.array, float):
    """
    :param y:
    :param tx:
    :return:
    """
    w_optim = np.linalg.lstsq(tx.T @ tx, tx.T @ y)
    return w_optim, MSELoss.loss(tx, y, w_optim)


def ridge_regression(y: np.array, tx: np.array, lambda_: float) -> (np.array, float):
    """
    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    regularizer_part = 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w_optim = np.linalg.lstsq(tx.T @ tx + regularizer_part, tx.T @ y)[0]
    return w_optim, RidgeLoss.loss(tx, y, w_optim, lambda_=lambda_)


def logistic_regression(
    y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float
) -> (np.array, float):
    """
    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """

    w = initial_w

    for n_iter in range(max_iters):
        grad = LogisticRegressionLoss.grad(tx, y, w)
        w -= gamma * grad

        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    # print(ws)
    # print(losses)

    return w, LogisticRegressionLoss.loss(tx, y, w)


def reg_logistic_regression(
    y: np.array,
    tx: np.array,
    lambda_: float,
    initial_w: np.array,
    max_iters: int,
    gamma: float,
    return_all_losses: bool = False,
) -> (np.array, float):
    """
    :param y:
    :param tx:
    :param lambda_:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    w = initial_w

    if return_all_losses:
        losses = [RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_)]

    for n_iter in tqdm(range(max_iters)):
        grad = RegLogisticRegressionLoss.grad(tx, y, w, lambda_=lambda_)
        w -= gamma * grad

        if return_all_losses:
            losses.append(RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_))
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    if return_all_losses:
        return w, RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_), losses
    else:
        return w, RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_)


def reg_logistic_regression_AGDR(
    y: np.array,
    tx: np.array,
    lambda_: float,
    initial_w: np.array,
    max_iters: int,
    gamma: float,
    return_all_losses: bool = False,
) -> (np.array, float):
    """
    :param y:
    :param tx:
    :param lambda_:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    w = v = initial_w
    t = 1

    if return_all_losses:
        losses = [RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_)]

    for _ in tqdm(range(max_iters)):
        next_w = v - gamma * RegLogisticRegressionLoss.grad(tx, y, v, lambda_=lambda_)

        if RegLogisticRegressionLoss.loss(
            tx, y, w, lambda_=lambda_
        ) < RegLogisticRegressionLoss.loss(tx, y, next_w, lambda_=lambda_):
            v = w
            t = 1
            next_w = v - gamma * RegLogisticRegressionLoss.grad(
                tx, y, v, lambda_=lambda_
            )

        next_t = (1 + (1 + 4 * t**2) ** 0.5) / 2
        next_v = next_w + ((t - 1) / next_t) * (next_w - w)

        w, v, t = next_w, next_v, next_t

        if return_all_losses:
            losses.append(RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_))

    if return_all_losses:
        return w, RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_), losses
    else:
        return w, RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_)
