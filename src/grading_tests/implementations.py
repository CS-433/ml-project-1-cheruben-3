import numpy as np

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
    ws = [initial_w]
    losses = [MSELoss.loss(tx, y, initial_w)]
    w = initial_w

    for n_iter in range(max_iters):
        grad = MSELoss.grad(tx, y, w)
        w -= gamma * grad

        ws.append(w)
        losses.append(MSELoss.loss(tx, y, w))
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    print(losses)

    return ws[-1], losses[-1]


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
    ws = [initial_w]
    losses = [MSELoss.loss(tx, y, initial_w)]
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(tx, y, batch_size=1, num_batches=1):
            grad = MSELoss.grad(tx_batch, y_batch, w)
            w = w - gamma * grad

            loss = MSELoss.loss(tx, y, w)

            ws.append(w)
            losses.append(loss)

            # print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]


def least_squares(y: np.array, tx: np.array) -> (np.array, float):
    """

    :param y:
    :param tx:
    :return:
    """
    # TODO: Do we use the inverse or solve the system instead? (in case the system is rank deficient)
    w_optim = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w_optim, MSELoss.loss(tx, y, w_optim)


def ridge_regression(y: np.array, tx: np.array, lambda_: float) -> (np.array, float):
    """

    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    regularizer_part = 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w_optim = np.linalg.inv(tx.T @ tx + regularizer_part) @ tx.T @ y
    print(RidgeLoss.loss(tx, y, w_optim, lambda_=lambda_))
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
    ws = [initial_w]
    losses = [LogisticRegressionLoss.loss(tx, y, initial_w)]
    w = initial_w

    for n_iter in range(max_iters):
        grad = LogisticRegressionLoss.grad(tx, y, w)
        w -= gamma * grad

        ws.append(w)
        losses.append(LogisticRegressionLoss.loss(tx, y, w))
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    # print(ws)
    # print(losses)

    return ws[-1], losses[-1]


def reg_logistic_regression(
    y: np.array,
    tx: np.array,
    lambda_: float,
    initial_w: np.array,
    max_iters: int,
    gamma: float,
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
    ws = [initial_w]
    losses = [RegLogisticRegressionLoss.loss(tx, y, initial_w, lambda_=lambda_)]
    w = initial_w

    for n_iter in range(max_iters):
        grad = RegLogisticRegressionLoss.grad(tx, y, w, lambda_=lambda_)
        w -= gamma * grad

        ws.append(w)
        losses.append(RegLogisticRegressionLoss.loss(tx, y, w, lambda_=lambda_))
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]
