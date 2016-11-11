import numpy as np

FEAT_DIM = 440


def activitynet_parsing(representation, labels, targets, mask, l2_norm=False,
                        std_scaling=True):
    """Reshape provided data

    Notes
    -----
    1. Assume labels are 0-indexed

    """
    # Remove nan if any
    idx_rm = np.isnan(representation).sum(axis=1) > 1
    if idx_rm.sum() > 1:
        print 'KaBOOM!!! Houston: dataset contains NaN values'
        representation, labels, targets, mask = (
            i[~idx_rm, ...] for i in [representation, labels, targets, mask])

    # Collect metadata
    n_instances, temporal_feat_dim = representation.shape
    n_categories = labels.max() + 2  # Extra class for background
    n_targets = targets.shape[1]
    if temporal_feat_dim % FEAT_DIM != 0:
        print 'Weird dataset!\nRunning at ur risk'

    # Shuffling
    idx_shuffle = np.random.permutation(n_instances)
    representation, labels, targets, mask = (
        i[idx_shuffle, ...] for i in [representation, labels, targets, mask])

    # representation
    # L2-normalization
    if l2_norm:
        l2_norm = np.expand_dims(np.sqrt((representation**2).sum(axis=1)), 1)
        l2_norm[l2_norm == 0] = 1.0
        representation = representation / l2_norm
    # Reshape tensor
    tensor_shape = (n_instances, -1, FEAT_DIM)
    X = representation.reshape(tensor_shape)
    # zero mean & unit variance per time step
    if std_scaling:
        mu, std = standard_scaling_1d_parameters(X)
        X = (X - mu) / std

    # target as sparse matrix
    Y_offsets = np.zeros((n_instances, n_targets * n_categories))
    for i in xrange(n_targets):
        idx_label_offset = n_targets * labels
        Y_offsets[(xrange(n_instances),
                   idx_label_offset + i)] = targets[:, i]

    # label matrix
    labels[mask < 0.5] = n_categories - 1
    Y_labels = np.eye(n_categories)[labels]
    return X, Y_labels, Y_offsets


def standard_scaling_1d_parameters(X, axis=0):
    """Compute mean and standard deviation for 3D tensor

    Parameters
    ----------
    X : ndarray
        3dim tensor
    axis : int
        dimension to scale

    Returns
    -------
    mean : ndarray
        3dim tensor
    std : ndarray
        3dim tensor

    """
    shape, sum_axes = list(X.shape), range(X.ndim)
    del sum_axes[axis]
    del shape[axis]
    sum_axes = tuple(sum_axes)
    reshape_for_output = [1] * X.ndim
    reshape_for_output[axis] = -1
    N = np.prod(shape)

    mean = X.sum(axis=sum_axes)
    mean = (mean/N).reshape(reshape_for_output)

    std = (X - mean)**2
    std = std.sum(axis=sum_axes)
    std = (std/(N - 1)).reshape(reshape_for_output)
    std[std == 0] = 1.0
    return mean, std
