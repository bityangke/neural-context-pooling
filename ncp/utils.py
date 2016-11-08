import numpy as np


def count_wraparound(start=0, end=0, step=1):
    """Infinite counter with wraparound

    Parameters
    ----------
    start : int
    end : int
    step : int

    Yield
    -----
    i : int

    """
    i = start
    if i == end:
        raise StopIteration()

    while True:
        yield i
        i += step
        if i >= end:
            i = i % end


def category_weighting(Y, eps=1e-12):
    """Weight each category according to its inverse frequency ratio

    Parameters
    ----------
    Y : ndarray
        2-dim binary label matrix
    eps : float
        tolerance value used to avoid division by zero. Adjust it such that it
        is less than 1 / max_freq

    Returns
    -------
    weight : ndarray
        1-dim vector with weights for each category

    """
    n_categories = Y.shape[1]
    discrete_labels = np.arange(n_categories).reshape((n_categories, 1))
    y = np.dot(Y, discrete_labels).astype(int)
    label_freq = np.bincount(y.reshape(-1), minlength=discrete_labels.size)
    inv_ratio = label_freq.max() * 1.0 / (label_freq + eps)
    inv_ratio[inv_ratio >= 1/eps] = 0.0
    weight = inv_ratio / inv_ratio.max()
    return weight
