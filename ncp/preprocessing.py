import numpy as np

FEAT_DIM = 440


def activitynet_parsing(representation, labels, targets, mask):
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

    # Reshape tensor
    tensor_shape = (n_instances, -1, FEAT_DIM)
    X = representation.reshape(tensor_shape)

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
