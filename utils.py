import numpy as np

def sigmoid(t):
    """ Implementation of sigmoid function.

    Sigmoid function scales all true values between 0 and 1.
    """
    return 1 / (1 + np.exp(- t))


def compute_loss(y, tx, w):
    """ Compute the loss for linear regression.

    Paras:
        y:  True value. The shape should be [n_samples,]
        tx: Feature matrix. The shape should be [n_samples, n_features]
        w:  Weights of linear regression.

    Returns:
        loss:   Computed loss. Float.
    """
    e = y.reshape(-1, 1) - tx.dot(w)
    loss = 1 / (2 * len(y)) * (e.T.dot(e))
    return loss


def compute_gradient(y, tx, w):
    """ Compute the gradients for linear regression.

    Paras:
        y:  True value. The shape should be [n_samples,]
        tx: Feature matrix. The shape should be [n_samples, n_features]
        w:  Weights of linear regression.

    Returns:
        grads:  Computed gradients. The shape is [n_features, 1]
    """
    e = y.reshape(-1, 1) - tx.dot(w)
    grads = - 1 / len(y) * tx.T.dot(e)
    return grads


def calculate_loss(y, tx, w):
    """ Calculate the cross entropy loss.

    Paras:
        y:  True value. The shape should be [n_samples,]
        tx: Feature matrix. The shape should be [n_samples, n_features]
        w:  Weights of linear regression.

    Returns:
        loss:   Computed loss. Float.

    """
    predicted_y = sigmoid(tx.dot(w))
    loss_vector = y * (np.log(predicted_y)) + (1 - y) * (np.log(1 - predicted_y))
    loss = - sum(loss_vector)
    return loss


def calculate_gradient(y, tx, w):
    """ Compute the gradients for cross entropy loss.

    Paras:
        y:  True value. The shape should be [n_samples,]
        tx: Feature matrix. The shape should be [n_samples, n_features]
        w:  Weights of linear regression.

    Returns:
        grads:  Computed gradients. The shape is [n_features, 1]
    """
    gradient = tx.T.dot(sigmoid(tx.dot(w)) - y)
    return gradient


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generators of batching.

    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def unpack_from_generator(mygenerator):
    for i in mygenerator:
        return i[0], i[1]


def train_test_split(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def accuracy_score(predicted, original):
    accuracy = (predicted.squeeze() == original.squeeze()).sum() / len(predicted)
    return accuracy


def standardize(x):
    avg = x.mean(axis=0)
    dev = x.std(axis=0)
    x = (x - avg) / dev
    return x


def findMissingOrErrorData(b):
    b = b + 999. * (b < -900.)
    return b


def transform_data(tx):
    X = findMissingOrErrorData(tx)
    # X = feature_engineering(X)
    X = standardize(X)
    return X