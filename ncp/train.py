import argparse
import inspect
import json
import os

import h5py
import numpy as np
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ProgbarLogger, ReduceLROnPlateau)
from keras.utils.io_utils import HDF5Matrix

from ncp import preprocessing
from ncp.model import (neural_context_model, neural_context_shallow_model,
                       set_learning_rate)
from ncp.utils import count_wraparound, category_weighting

HDF5_DATASETS = ['Representation', 'Labels', 'Targets', 'RegressionMask']
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
    dmem_parser = p.add_mutually_exclusive_group(required=False)
    dmem_parser.add_argument('-dim', '--in-memory', action='store_true',
                             help='Dataset fits in memory')
    dmem_parser.add_argument('-ndim', '--no-in-memory',
                             dest='in_memory', action='store_false')
    p.add_argument('-dbz', '--batch-size', nargs='+', default=[1000], type=int,
                   help=('Batch size. Pass two values to different batch size '
                         'in [training, validation].'))
    p.add_argument('-dts', '--train-samples', default=45000, type=int,
                   help='Number of training samples')
    p.add_argument('-dvs', '--validation-split', default=0.2, type=float,
                   help=('Ratio of samples (last) from training used as '
                         'validation'))
    zmuv_parser = p.add_mutually_exclusive_group(required=False)
    zmuv_parser.add_argument('-dss', '--std-scaling', action='store_true',
                             help='Perform zero-mean and unit variance')
    zmuv_parser.add_argument('-ndss', '--no-std-scaling',
                             dest='std_scaling', action='store_false')
    zmuv_parser.set_defaults(std_scaling=False)
    # Architecture
    p.add_argument('-af', '--arch-file', nargs='?', default=JSON_ARCH_EXAMPLE,
                   help='JSON-file to modify architecture')
    shallow_parser = p.add_mutually_exclusive_group(required=False)
    shallow_parser.add_argument('-asm', '--arch-shallow', action='store_true',
                                help='Dataset fits in memory')
    shallow_parser.add_argument('-nasm', '--no-arch-shallow',
                                dest='arch_shallow', action='store_false')
    shallow_parser.set_defaults(arch_shallow=False)
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


def data_generator(X, Y_prob, Y_offsets, batch_size, n_samples=0):
    """Data feeder

    Parameters
    ----------
    X : ndarray, h5py, pandas, etc.
    Y_prob : ndarray, h5py, pandas, etc.
    Y_offsets : ndarray, h5py, pandas, etc.
    batch_size : int
    n_samples : int, optional

    """
    counter = count_wraparound(end=n_samples)
    while True:
        indexes = [counter.next() for i in xrange(batch_size)]
        yield {'context_over_time': X[indexes, ...],
               'output_prob': Y_prob[indexes, ...],
               'output_offsets': Y_offsets[indexes, ...]}


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


def load_dataset_by_chunks(filename, batch_size, train_samples=None,
                           validation_split=0.25, hdf5_datasets=HDF5_DATASETS):
    """Load dataset in memory from HDF5-file

    Parameters
    ----------
    filename : str
        fullpath of hdf5 file
    batch_size : iterable of int
        Number of instances per batch (training, validation)
    train_samples : int
        Number or training samples
    validation_split : float
        Number/ratio of samples to use as validation set

    Returns
    -------
    train_generator : generator
    val_generator : generator
    samples_per_training : int
    samples_per_validation : int
    metadata : tuple
        Extra info about dataset (num-categories, receptive-field)

    """
    if not os.path.exists(filename):
        raise IOError('Unexistent file {}'.format(filename))
    if len(batch_size) > 1:
        train_batch_size, val_batch_size = batch_size[0:1]
    else:
        train_batch_size = val_batch_size = batch_size[0]

    train_data = [HDF5Matrix(filename, i, end=train_samples)
                  for i in hdf5_datasets]
    train_data_size = train_data[0].shape
    num_categories = train_data[1].shape[1]
    train_instances, receptive_field = train_data_size[0], train_data_size[1:]
    samples_per_training = int(np.ceil(train_instances / train_batch_size))
    X_trn, Y_prob_trn, Y_offsets_trn = train_data
    train_generator = data_generator(X_trn, Y_prob_trn, Y_offsets_trn,
                                     batch_size, train_instances)

    if validation_split > 1:
        validation_split = int(validation_split)
    else:
        validation_split = int(np.ceil((train_instances * validation_split)))
    validation_data = [HDF5Matrix(filename, i, start=validation_split)
                       for i in hdf5_datasets]
    val_instances = validation_data[0].shape[0]
    samples_per_validation = int(val_instances / val_batch_size)
    X_val, Y_prob_val, Y_offsets_val = validation_data
    validation_generator = data_generator(X_val, Y_prob_val, Y_offsets_val,
                                          val_batch_size, val_instances)

    metadata = (num_categories, receptive_field)
    argout = (train_generator, validation_generator, samples_per_training,
              samples_per_validation, metadata)
    return argout


def load_dataset_in_memory(filename, hdf5_datasets=HDF5_DATASETS,
                           std_scaling=True):
    """Load dataset in memory from HDF5-file

    Parameters
    ----------
    filename : str
        fullpath of hdf5 file
    std_scaling : bool
        Enable zero mean and unit variance scaling

    Returns
    -------
    X : ndarray
    Y_labels : ndarray
    Y_offsets : ndarray
    train_samples : int
    metadata : tuple
        Extra info about dataset (num-categories, receptive-field)

    """
    if not os.path.exists(filename):
        raise IOError('Unexistent file {}'.format(filename))

    fid = h5py.File(filename, 'r')
    dataset = [fid[i][...] for i in hdf5_datasets]

    X, Y_labels, Y_offsets = preprocessing.activitynet_parsing(
        *dataset, std_scaling=std_scaling)
    metadata = Y_labels.shape[1], X.shape[1::]
    return X, Y_labels, Y_offsets, metadata


def main(dataset_file, in_memory, batch_size, train_samples, validation_split,
         std_scaling, arch_file, arch_shallow, alpha, lr_start, lr_gain,
         lr_patience, stop_patience, max_epochs, output_dir, verbosity,
         **kwargs):
    # Check output-folder
    initialize_logging(output_dir, inspect.currentframe())

    # Read architecture
    with open(arch_file) as fid:
        arch_prm = json.load(fid)

    # Load dataset
    if in_memory:
        dataset_tuple = load_dataset_in_memory(
            dataset_file, std_scaling=std_scaling)
        X, Y_labels, Y_offsets, metadata = dataset_tuple
        # Small piece of dataset
        train_samples = min(train_samples, X.shape[0])
        X, Y_labels, Y_offsets = (X[0:train_samples, ...],
                                  Y_labels[0:train_samples, ...],
                                  Y_offsets[0:train_samples, ...])
        # weight classes
        weights = category_weighting(Y_labels)
        weights = weights / weights.mean()
        class_weight = dict(zip(xrange(weights.size), weights))
        # Set unique batch_size
        batch_size = batch_size[0]
        if verbosity > 0:
            print 'Ignoring validation batch size. Keras limitation.'
    else:
        dataset_tuple = load_dataset_by_chunks(
            dataset_file, batch_size, train_samples, validation_split)
        (train_generator, validation_generator,
         samples_per_training, samples_per_validation,
         metadata) = dataset_tuple
    num_categories, receptive_field = metadata

    # Model instantiation
    if arch_shallow:
        model = neural_context_shallow_model(num_categories, receptive_field)
        model.compile(optimizer='rmsprop',
                      loss={'output_prob': 'hinge',
                            'output_offsets': 'mse'},
                      loss_weights={'output_prob': 1.,
                                    'output_offsets': alpha},
                      metrics={'output_prob': 'categorical_accuracy',
                               'output_offsets': 'mean_absolute_error'})
    else:
        model = neural_context_model(num_categories, receptive_field, arch_prm)
        model.compile(optimizer='rmsprop',
                      loss={'output_prob': 'categorical_crossentropy',
                            'output_offsets': 'mse'},
                      loss_weights={'output_prob': 1.,
                                    'output_offsets': alpha},
                      metrics={'output_prob': 'categorical_accuracy',
                               'output_offsets': 'mean_absolute_error'})
    set_learning_rate(model, lr_start)

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

    if verbosity > 1:
        lst_callbacks += [ProgbarLogger()]

    # Training
    if verbosity > 0:
        print 'Optimization begins'
    if in_memory:
        model.fit(X, {'output_prob': Y_labels, 'output_offsets': Y_offsets},
                  nb_epoch=max_epochs, batch_size=batch_size, shuffle=True,
                  validation_split=validation_split, callbacks=lst_callbacks,
                  class_weight=class_weight)
    else:
        queue_size, max_workers = 1, 4
        model.fit_generator(
            train_generator, nb_epoch=max_epochs, callbacks=lst_callbacks,
            samples_per_epoch=samples_per_training,
            validation_data=validation_generator,
            nb_val_samples=samples_per_validation,
            max_q_size=queue_size, nb_worker=max_workers)
    if verbosity > 0:
        print 'Optimization finished'
    return None


if __name__ == '__main__':
    p = input_parser()
    main(**vars(p.parse_args()))
