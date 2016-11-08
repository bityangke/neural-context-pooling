import numpy as np

FEAT_DIM = 440


def activitynet_parsing(representation, labels, targets, mask):
    """Reshape provided data

    Notes
    -----
    1. Assume labels are 0-indexed

    """
    n_instances, temporal_feat_dim = representation.shape
    n_categories = labels.max() + 2  # Extra class for background
    n_targets = targets.shape[1]
    if temporal_feat_dim % FEAT_DIM != 0:
        print 'Weird dataset!\nRunning at ur risk'

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
