import numpy as np

FEAT_DIM = 440


def activitynet_parsing(representation, labels, targets):
    """Reshape provided data
    """
    n_instances, temporal_feat_dim = representation.shape
    n_categories = labels.max() + 1
    n_targets = targets.shape[1]
    if temporal_feat_dim % FEAT_DIM != 0:
        print 'Weird dataset!\nRunning at ur risk'

    # Focus on fine scale in the meantime
    fine_scale_start_idx = 3 * FEAT_DIM
    tensor_shape = (n_instances, -1, FEAT_DIM)
    X = representation[:, fine_scale_start_idx::].reshape(tensor_shape)

    # target as sparse matrix
    Y_offsets = np.zeros((n_instances, n_targets * n_categories))
    for i in xrange(n_targets):
        idx_label_offset = n_targets * labels
        Y_offsets[(xrange(n_instances),
                   idx_label_offset + i)] = targets[:, i]

    # label matrix
    Y_labels = np.eye(n_categories)[labels]
    return X, Y_labels, Y_offsets
