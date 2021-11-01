import numpy.ma as ma
import joblib
from proj1_helpers import *
import numpy as np
from enum import Enum
import time
from datetime import datetime, timedelta

from utils import *


class FuctionType(Enum):
    LEASTSQURE_SD = 1
    LEASTSQURE_SGD = 2
    LEASTSQUERE_NORMAL = 3
    RIDGE_REGRESSION = 4
    LOGIST_REGRESSION = 5
    REGULAR_LOGIST_REGRESSION = 6
    LIGHTGBM = 7
    DNN = 8

# 01 - Implementation of linear regression using gradient descent

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Gradient descent algorithm.

    Paras:
        y:          True value. The shape should be [n_samples,]
        tx:         Feature matrix. The shape should be [n_samples, n_features]
        initial_w:  Initial weights.
        max_iters:  Maximum iterations.
        gamma:      Learning rate.

    Returns:
        lass_w:     Updated weights.
        last_loss:  Final loss.
    """
    print('Running least_squares_GD...')
    # Define parameters to store w and loss
    ws = [initial_w.reshape(-1, 1)]
    losses = []
    w = initial_w.reshape(-1, 1)
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient

        ws.append(w)
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    last_w = ws[-1]
    last_loss = losses[-1]
    return last_w, last_loss

# 02 - Implementation of linear regression using stochastic gradient descent

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """ Stochastic gradient descent algorithm.

    Paras:
        y:          True value. The shape should be [n_samples,]
        tx:         Feature matrix. The shape should be [n_samples, n_features]
        initial_w:  Initial weights.
        max_iters:  Maximum iterations.
        gamma:      Learning rate.
    
    Returns:
        lass_w:     Updated weights.
        last_loss:  Final loss.
    """
    print('Running least_squares_SGD...')
    sgd_generator = batch_iter(y, tx, batch_size)
    sgd_y, sgd_tx = unpack_from_generator(sgd_generator)

    ws = [initial_w.reshape(-1, 1)]
    losses = []
    w = initial_w.reshape(-1, 1)
    for n_iter in range(max_iters):
        gradient = compute_gradient(sgd_y, sgd_tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]

# 03 - Implementation of least squares regression using normal equations.

def least_squares(y, tx):
    """ Calculate the lease square errors.

    Paras:
        y:  True value. The shape should be [n_samples,]
        tx: Feature matrix. The shape should be [n_samples, n_features]

    Returns:
        w:      Computed weights.
        loss:   Computed loss.
    """
    print('Running least_squares...')
    y = y.reshape(-1, 1)
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

# 04 - Implementation of ridge regression using normal equation

def ridge_regression(y, tx, lambda_):
    """ Implementation of ridge regression by solving normal equations.

    Paras:
        y:          True value. The shape should be [n_samples,]
        tx:         Feature matrix. The shape should be [n_samples, n_features]
        lambda_:    

    Returns:
        w:      Computed weights.
        loss:   Computed loss.
    """
    print('Running ridge_regression...')
    w = np.linalg.inv(tx.T.dot(tx) + lambda_ * 2 * len(y) *
                      np.identity(len(tx.T.dot(tx)))).dot(tx.T).dot(y)
    ridge_loss = compute_loss(y, tx, w) + lambda_ * (np.linalg.norm(w) ** 2)
    loss = compute_loss(y, tx, w)
    return w, loss

# 05 - Implementation of logistic regression

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression.

    Paras:
        y:          True value. The shape should be [n_samples,]
        tx:         Feature matrix. The shape should be [n_samples, n_features]
        initial_w:  Initial weights.
        max_iters:  Maximum iterations.
        gamma:      Learning rate.
    
    Returns:
        lass_w:     Updated weights.
        last_loss:  Final loss.
    """
    print('Running logistic_regression...')
    # init parameters
    losses = []
    ws = [initial_w]
    w = initial_w
    # start the logistic regression
    for n_iter in range(max_iters):
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w)

        w = w - gamma * gradient

        losses.append(loss)
        ws.append(w)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    last_w = ws[-1]
    last_loss = losses[-1]

    return last_w, last_loss

# 06 - Implementation of regularized logistic regression using GD or SGD

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression.

    Paras:
        y:          True value. The shape should be [n_samples,]
        tx:         Feature matrix. The shape should be [n_samples, n_features]
        lambda_:    Coefficient of L2 regularization.
        initial_w:  Initial weights.
        max_iters:  Maximum iterations.
        gamma:      Learning rate.
    
    Returns:
        lass_w:     Updated weights.
        last_loss:  Final loss.
    """
    print('Running reg_logistic_regression...')
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = calculate_loss(y, tx, w) + lambda_ * np.linalg.norm(w) ** 2
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w

        w = w - gradient * gamma

        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    last_w = ws[-1]
    last_loss = losses[-1]

    return last_w, last_loss


def run_little_minions(minion_Type, y, tx, y_test, x_test, gamma, lambda_, max_iters):
    if minion_Type == 1:
        w_train, loss_train = least_squares_GD(
            y, tx, initial_w, max_iters, gamma)
    elif minion_Type == 2:
        w_train, loss_train = least_squares_SGD(
            y, tx, initial_w, batch_size, max_iters, gamma)
    elif minion_Type == 3:
        w_train, loss_train = least_squares(y, tx)
    elif minion_Type == 4:
        w_train, loss_train = ridge_regression(y, tx, lambda_)
    elif minion_Type == 5:
        w_train, loss_train = logistic_regression(
            y, tx, initial_w, max_iters, gamma)
    elif minion_Type == 6:
        w_train, loss_train = reg_logistic_regression(
            y, tx, lambda_, initial_w, max_iters, gamma)
    return w_train


if __name__ == '__main__':
    # Read train data
    # TODO: download train data and supply path here
    DATA_TRAIN_PATH = './project_1/train.csv'
    y, tx, ids = load_csv_data(DATA_TRAIN_PATH)
    # parameters to manipulate
    X = transform_data(tx)
    ratio = 0.9
    x_train, x_test, y_train, y_test = train_test_split(X, y, ratio)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Start running minions
    initial_w = np.zeros((tx.shape[1], 1))
    max_iters = 100
    gamma = 0.01
    lambda_ = 0.1
    batch_size = 1000

    startDT = datetime.now()
    w_train = run_little_minions(
        4,
        y_train,
        x_train,
        y_test,
        x_test,
        gamma,
        lambda_,
        max_iters
    )
    endDT = datetime.now()
    print("Training time:", str(endDT - startDT))

    y_pred_test = predict_labels(w_train, x_test)

    accuracy = accuracy_score(y_pred_test, y_test)
    print('Accuracy: {}'.format(accuracy))

    # TODO: download train data and supply path here
    DATA_TEST_PATH = './project_1/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    # TODO: fill in desired name of output file for submission
    OUTPUT_PATH = './project_1/output.csv'
    y_pred = predict_labels(w_train, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
