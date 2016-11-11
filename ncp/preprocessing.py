import numpy as np

FEAT_DIM = 440


def activitynet_parsing(representation, labels, targets, mask, shuffle=False,
                        l2_norm=False, reshape=True, std_scaling=True):
    """Pre-processing done in ActivityNet data

    Notes
    -----
    1. Assume labels are 0-indexed

    """
    train_split = 0.7
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
    if shuffle:
        idx_shuffle = np.random.permutation(n_instances)
        representation, labels, targets, mask = (
            i[idx_shuffle, ...] for i in [representation, labels,
                                          targets, mask])

    # representation
    # L2-normalization
    if l2_norm:
        l2_norm = np.expand_dims(np.sqrt((representation**2).sum(axis=1)), 1)
        l2_norm[l2_norm == 0] = 1.0
        representation = representation / l2_norm
    # Reshape tensor
    if reshape:
        tensor_shape = (n_instances, -1, FEAT_DIM)
        X = representation.reshape(tensor_shape)
    # zero mean & unit variance per time step
    if std_scaling and reshape:
        up_to = int(train_split * n_instances)
        mu, std = standard_scaling_1d_parameters(X[0:up_to, ...], axis=1)
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


def held_out(y, lst_arrays, pctg=0.3, rng_seed=259810):
    """Reorganize lst_array such that examples are mutually exclusive and
       validation is balanced

    Parameters
    ----------
    y : ndarray
        Label vector (1d-array)
    lst_array : list
    pctg : float
        percentage of videos to held-out

    """
    rng = np.random.RandomState(rng_seed)
    n_instances = y.size
    vid_idx = infer_index_videos(y)
    last_idx = np.hstack([vid_idx[1:], [n_instances]])
    y_vid = y[vid_idx]
    n_classes = y.max() + 1

    # Held-out "pctg" of videos from a particular class
    nvid_per_class = np.bincount(y_vid, minlength=n_classes)
    nvid_held_out = (pctg * nvid_per_class).astype(int)
    idx_freeze, idx_held = [None] * n_classes, [None] * n_classes
    for i in xrange(n_classes):
        n_held = nvid_held_out[i]
        idx_shuffle = (y_vid == i).nonzero()[0]
        rng.shuffle(idx_shuffle)
        idx_held[i] = idx_shuffle[-n_held:]
        idx_freeze[i] = idx_shuffle[0:-n_held]

    # Create top-bottom indexes
    idx_top = np.hstack([range(vid_idx[i], last_idx[i])
                         for i in np.hstack(idx_freeze)])
    rng.shuffle(idx_top)
    # Most bottom part will have uniform labels in terms of videos
    min_ = np.min([i.size for i in idx_held])
    upper_bottom = np.hstack([i[min_:] for i in idx_held])
    idx_ubottom = np.hstack([range(vid_idx[i], last_idx[i])
                             for i in np.hstack(upper_bottom)])
    rng.shuffle(idx_ubottom)

    most_bottom = np.hstack([i[:min_] for i in idx_held])
    idx_bottom = np.hstack([range(vid_idx[i], last_idx[i])
                            for i in np.hstack(most_bottom)])
    new_arrange = np.hstack([idx_top, idx_ubottom, idx_bottom])
    print 'Balanced validation split:', idx_bottom.size * 1.0 / n_instances

    return new_arrange


def infer_index_videos(label_vector):
    """Return index of unique video-ids

    Parameters
    ----------
    label_vector : ndarray

    Notes
    -----
        Infer video-ids by assumming that features were dumpped serially by
        videos.

    """
    detect_change = label_vector[1:] - label_vector[:-1]
    idx = np.nonzero(np.hstack([[1], detect_change]))[0]
    return idx


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
