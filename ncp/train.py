import argparse
import inspect
import json
import os

import numpy as np
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ProgbarLogger, ReduceLROnPlateau)

from ncp.model import neural_context_model, set_learning_rate

JSON_ARCH_EXAMPLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'examples',
    'arch.json')


def input_parser(p=None):
    """Argument parser for training NCP model
    """
    if p is None:
        p = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Batch/Dataset parameters
    p.add_argument('-dh5', '--dataset-file', default='awesome-dataset.hdf5',
                   help='HDF5-file with dataset')
    p.add_argument('-dbz', '--batch-size', default=1000, type=int,
                   help='Batch size')
    p.add_argument('-dts', '--train-samples', default=45000, type=int,
                   help='Number of training samples')
    p.add_argument('-dvs', '--validation-split', default=0.2, type=float,
                   help=('Ratio of samples (last) from training used as '
                         'validation'))
    # Architecture
    p.add_argument('-af', '--arch-file', nargs='?', default=JSON_ARCH_EXAMPLE,
                   help='JSON-file to modify architecture')
    # Optimization parameters
    p.add_argument('-oa', '--alpha', default=0.2, type=float,
                   help='Weight contribution btw refinement and prediction')
    p.add_argument('-olri', '--lr-start', default=1e-1, type=float,
                   help='Initial learning rate')
    p.add_argument('-olrf', '--lr-gain', default=0.1, type=float,
                   help='Gain learning rate')
    p.add_argument('-olrp', '--lr-patience', default=5, type=int,
                   help=('Number of epochs without improvement before '
                         'decreasing learning-rate'))
    p.add_argument('-oesp', '--stop-patience', default=13, type=int,
                   help=('Number of epochs without improvement before '
                         'stopping DNN-optimization'))
    p.add_argument('-ome', '--max-epochs', default=50, type=int,
                   help='Max number of training epochs during CNN update')
    # Logging and outputs
    p.add_argument('-rod', '--output-dir', default='.',
                   help='Fullpath of dir to allocate log and outputs')
    p.add_argument('-v', '--verbosity', default=0, type=int)
    return p


def initialize_logging(output_dir, frame,
                       remove_keys=['verbosity', 'args', 'kwargs']):
    """Create output_dir and save arguments as JSON file

    Parameters
    ----------
    output_dir : str
        Fullpath of dirname
    frame : inspect.frame
        frame info
    remove_keys : list
        string of var-names to remove from frame

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _, _, _, values = inspect.getargvalues(frame)
    for i in remove_keys:
        values.pop(i, None)
    argfile = os.path.join(output_dir, 'arguments.json')
    with open(argfile, 'w') as fid:
        json.dump(values, fid, sort_keys=True, indent=4)


def load_dataset(filename, extra_args):
    """Load dataset from HDF5-file

    Parameters
    ----------
    filename : str
        fullpath of hdf5 file

    Returns
    -------
    X : ndarray
        Features, [num_instances, t_steps, feat_dim]
    Y_labels : ndarray
        Label matrix [num_instances, n_categories]
    Y_offsets : ndarray
        Target offsets [num_instances, n_categories * 2]

    """
    train_samples, num_classes = extra_args
    X = np.random.rand(train_samples, 10, 512)
    Y_labels = np.eye(num_classes)[np.random.randint(0, num_classes,
                                                     train_samples), :]
    Y_offsets = np.random.rand(train_samples, 2 * num_classes)
    return X, Y_labels, Y_offsets


def main(dataset_file, batch_size, train_samples, validation_split, arch_file,
         alpha, lr_start, lr_gain, lr_patience, stop_patience, max_epochs,
         output_dir, verbosity, **kwargs):
    initialize_logging(output_dir, inspect.currentframe())
    # Read architecture
    with open(arch_file) as fid:
        arch_prm = json.load(fid)
    # Load dataset
    num_classes = 20
    X, Y_labels, Y_offsets = load_dataset(dataset_file,
                                          (train_samples, num_classes))
    num_classes = Y_labels.shape[1]
    train_samples = min(train_samples, X.shape[0])
    receptive_field = X.shape[1::]

    # Model instantiation
    model = neural_context_model(num_classes, receptive_field, arch_prm)
    model.compile(optimizer='rmsprop',
                  loss={'output_prob': 'categorical_crossentropy',
                        'output_offsets': 'mse'},
                  loss_weights={'output_prob': 1.,
                                'output_offsets': alpha},
                  metrics={'output_prob': 'categorical_accuracy',
                           'output_offsets': 'mean_absolute_error'})
    set_learning_rate(model, lr_start)
    # Initialize model weights: Do you need that?

    # Callbacks instantiation
    var_monitored = 'val_output_prob_categorical_accuracy'
    lst_callbacks = []

    lst_callbacks += [
        ReduceLROnPlateau(
            monitor=var_monitored, factor=lr_gain, patience=lr_patience,
            epsilon=1e-8, min_lr=1e-8, verbose=1)]

    lst_callbacks += [
        EarlyStopping(monitor=var_monitored, patience=stop_patience,
                      verbose=1)]

    filepath = os.path.join(output_dir, 'weights.hdf5')
    lst_callbacks += [
        ModelCheckpoint(
            filepath, monitor=var_monitored, verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto')]

    trainlog = os.path.join(output_dir, 'train.log')
    lst_callbacks += [
        CSVLogger(trainlog, separator=' ')]

    if verbosity > 0:
        lst_callbacks += [ProgbarLogger()]

    # Training
    if verbosity > 0:
        print 'Optimization begins'
    X, Y_labels, Y_offsets = (X[0:train_samples, ...],
                              Y_labels[0:train_samples, ...],
                              Y_offsets[0:train_samples, ...])
    model.fit(X, {'output_prob': Y_labels, 'output_offsets': Y_offsets},
              nb_epoch=max_epochs, batch_size=batch_size, shuffle=False,
              validation_split=validation_split, callbacks=lst_callbacks)

    if verbosity > 0:
        print 'Optimization finished'


if __name__ == '__main__':
    p = input_parser()
    main(**vars(p.parse_args()))
