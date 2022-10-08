import numpy as np

from helpers import batch_iter
from metrics import MSELoss, RidgeLoss

from tqdm import tqdm


def least_squares_GD(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float) \
        -> (np.array, float):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in tqdm(range(max_iters)):
        loss, grad = MSELoss.eval(tx, y, w)
        w -= gamma * grad

        ws.append(w)
        losses.append(loss)
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]


def least_squares_SGD(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float) \
        -> (np.array, float):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in tqdm(range(max_iters)):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = MSELoss.grad(tx_batch, y_batch, w)
            loss = MSELoss.loss(tx, y, w)
            w = w - gamma * grad

            ws.append(w)
            losses.append(loss)

            # print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]


def least_squares(y: np.array, tx: np.array) \
        -> (np.array, float):
    # TODO: Do we use the inverse or solve the system instead? (in case the system is rank deficient)
    w_optim = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w_optim, MSELoss.loss(tx, y, w_optim)


def ridge_regression(y: np.array, tx: np.array, lambda_: float) \
        -> (np.array, float):
    regularizer_part = 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w_optim = np.linalg.inv(tx.T @ tx + regularizer_part) @ tx.T @ y
    return w_optim, RidgeLoss.loss(tx, y, w_optim, lambda_=lambda_)


def logistic_regression(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float) \
        -> (np.array, float):
    pass


def reg_logistic_regression(y: np.array, tx: np.array, lambda_: float, initial_w: np.array, max_iters: int, gamma: float
                            ) -> (np.array, float):
    pass
