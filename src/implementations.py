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
    """Performs gradient descent on a linear regression model to find the optimal parameters and final loss

    Args:
        y: 2d numpy array where the rows represent a sample the column value represents the label of the sample
        tx: 2d numpy array where the rows represent a sample the column values represent the features of the sample
        initial_w: numpy array representing the weights from which to start optimization
        max_iters: the number of iterations to run gradient descent for
        gamma: the learning rate

    Returns:
        2-tuple of the following:
        - The final weight after max_iters iterations
        - The loss of the final weights on (tx, y)

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
    """Performs STOCHASTIC gradient descent on a linear regression model to find the optimal parameters and final loss

    Args:
        y: 2d numpy array where the rows represent a sample the column value represents the label of the sample
        tx: 2d numpy array where the rows represent a sample the column values represent the features of the sample
        initial_w: numpy array representing the weights from which to start optimization
        max_iters: the number of iterations to run gradient descent for
        gamma: the learning rate

    Returns:
        2-tuple of the following:
        - The final weight after max_iters iterations
        - The loss of the final weights on (tx, y)

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
    """Returns the least-squares approximation of a linear regression model to find the optimal parameters and final loss

    Args:
        y: 2d numpy array where the rows represent a sample the column value represents the label of the sample
        tx: 2d numpy array where the rows represent a sample the column values represent the features of the sample

    Returns:
        2-tuple of the following:
        - The final weight after max_iters iterations
        - The loss of the final weights on (tx, y)

    """
    w_optim = np.linalg.lstsq(tx.T @ tx, tx.T @ y, rcond=None)[0]
    return w_optim, MSELoss.loss(tx, y, w_optim)


def ridge_regression(y: np.array, tx: np.array, lambda_: float) -> (np.array, float):
    """Returns the least-squares approximation of a REGULARIZED linear regression model to find the optimal parameters 
    and final loss (note that lambda_ could be 0)

    Args:
        y: 2d numpy array where the rows represent a sample the column value represents the label of the sample
        tx: 2d numpy array where the rows represent a sample the column values represent the features of the sample
        lambda_: the weight on the regularization term of the loss (higher = more regularization)

    Returns:
        2-tuple of the following:
        - The final weight after max_iters iterations
        - The loss of the final weights on (tx, y)

    """
    regularizer_part = 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w_optim = np.linalg.lstsq(tx.T @ tx + regularizer_part, tx.T @ y, rcond=None)[0]
    return w_optim, RidgeLoss.loss(tx, y, w_optim, lambda_=lambda_)


def logistic_regression(
    y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float
) -> (np.array, float):
    """Performs gradient descent on a logistic regression model to find the optimal parameters and final loss

    Args:
        y: 2d numpy array where the rows represent a sample the column value represents the label of the sample
        tx: 2d numpy array where the rows represent a sample the column values represent the features of the sample
        initial_w: numpy array representing the weights from which to start optimization
        max_iters: the number of iterations to run gradient descent for
        gamma: the learning rate

    Returns:
        2-tuple of the following:
        - The final weight after max_iters iterations
        - The loss of the final weights on (tx, y)

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
    """Performs gradient descent on a REGULARIZED logistic regression model to find the optimal parameters and final
    loss (note that lambda_ could be 0)

    Args:
        y: 2d numpy array where the rows represent a sample the column value represents the label of the sample
        tx: 2d numpy array where the rows represent a sample the column values represent the features of the sample
        lambda_: the weight on the regularization term of the loss (higher = more regularization)
        initial_w: numpy array representing the weights from which to start optimization
        max_iters: the number of iterations to run gradient descent for
        gamma: the learning rate
        return_all_losses: boolean for the purpose of tracking the history of the loss, useful for experimenting

    Returns:
        IF return_all_losses is FALSE:
            2-tuple of the following:
            - The final weight after max_iters iterations
            - The loss of the final weights on (tx, y)
        IF return_all_losses is TRUE:
            3-tuple of the following:
            - The final weight after max_iters iterations
            - The loss of the final weights on (tx, y)
            - The history of the loss on each iteration

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
    """Performs ACCELERATED gradient descent WITH RESTART on a REGULARIZED logistic regression model to find the optimal
    parameters and final loss (note that lambda_ could be 0)

    -> On a high-level, this gradient descent algorithm keeps some sort of "momentum" to boost the algorithm in the
    right direction and resets this momentum when it goes in a bad direction

    Args:
        y: 2d numpy array where the rows represent a sample the column value represents the label of the sample
        tx: 2d numpy array where the rows represent a sample the column values represent the features of the sample
        lambda_: the weight on the regularization term of the loss (higher = more regularization)
        initial_w: numpy array representing the weights from which to start optimization
        max_iters: the number of iterations to run gradient descent for
        gamma: the learning rate
        return_all_losses: boolean for the purpose of tracking the history of the loss, useful for experimenting

    Returns:
        IF return_all_losses is FALSE:
            2-tuple of the following:
            - The final weight after max_iters iterations
            - The loss of the final weights on (tx, y)
        IF return_all_losses is TRUE:
            3-tuple of the following:
            - The final weight after max_iters iterations
            - The loss of the final weights on (tx, y)
            - The history of the loss on each iteration

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
