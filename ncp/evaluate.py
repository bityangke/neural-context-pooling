import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from ncp.model import neural_context_model, neural_context_shallow_model
from ncp.train import load_dataset_in_memory

HDF5_DATASETS = ['Representation', 'Labels', 'Targets', 'RegressionMask']


def input_parser(p=None):
    """Argument parser for training NCP model
    """
    if p is None:
        p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # Model file
    p.add_argument('-md', '--modelfile', default='non-existent.hdf5',
                   help='HDF5-file with weights from model to evaluate')
    # Batch/Dataset parameters
    p.add_argument('-dh5', '--dataset-file', default='awesome-dataset.hdf5',
                   help='HDF5-file with dataset')
    p.add_argument('-dis', '--idx-start', default=0, type=int,
                   help='Index to start reading row from HDF5')
    p.add_argument('-die', '--idx-end', default=-1, type=int,
                   help='Index to stop reading row from HDF5')
    p.add_argument('-dbz', '--batch-size', default=5000, type=int,
                   help='Batch size')
    # Data preprocessing
    l2n_parser = p.add_mutually_exclusive_group(required=False)
    l2n_parser.add_argument('-dpl2', '--l2-norm', action='store_true',
                            help='l2 normalization along rows')
    l2n_parser.add_argument('-ndpl2', '--no-l2-norm',
                            dest='l2_norm', action='store_false')
    l2n_parser.set_defaults(l2_norm=False)
    zmuv_parser = p.add_mutually_exclusive_group(required=False)
    zmuv_parser.add_argument('-dss', '--std-scaling', action='store_true',
                             help='Perform zero-mean and unit variance')
    zmuv_parser.add_argument('-ndss', '--no-std-scaling',
                             dest='std_scaling', action='store_false')
    zmuv_parser.set_defaults(std_scaling=False)
    # Architecture
    p.add_argument('-af', '--arch-file', nargs='?', default='not-exist.json',
                   help='JSON-file to modify architecture')
    shallow_parser = p.add_mutually_exclusive_group(required=False)
    shallow_parser.add_argument('-asm', '--arch-shallow', action='store_true',
                                help='Dataset fits in memory')
    shallow_parser.add_argument('-nasm', '--no-arch-shallow',
                                dest='arch_shallow', action='store_false')
    shallow_parser.set_defaults(arch_shallow=False)
    # Logging and outputs
    p.add_argument('-v', '--verbosity', default=0, type=int)
    return p


def main(modelfile, dataset_file, idx_start, idx_end, batch_size, l2_norm,
         std_scaling, arch_file, arch_shallow, verbosity, **kwargs):
    # Read architecture
    if not arch_shallow:
        if not os.path.isfile(arch_file):
            raise IOError('Unexistent arch JSON {}'.format(arch_file))
        with open(arch_file) as fid:
            arch_prm = json.load(fid)

    # Load dataset
    dataset_tuple = load_dataset_in_memory(
        dataset_file, l2_norm=l2_norm, std_scaling=std_scaling)
    X, Y_labels, Y_offsets, metadata = dataset_tuple
    # Take validation set
    if idx_end == -1:
        idx_end = X.shape[0]
    X, Y_labels, Y_offsets = (i[idx_start:idx_end, ...]
                              for i in [X, Y_labels, Y_offsets])
    num_categories, receptive_field = metadata

    # Model instantiation
    # loss_weights (balance btw classification and regression)
    if arch_shallow:
        model = neural_context_shallow_model(num_categories, receptive_field)
    else:
        model = neural_context_model(num_categories, receptive_field, arch_prm)

    if verbosity > 0:
        print 'Compiling model'
    model.compile(optimizer='rmsprop',
                  loss={'output_prob': 'categorical_crossentropy',
                        'output_offsets': 'mse'})
    if verbosity > 0:
        print 'Loading weights'
    model.load_weights(modelfile)

    # Evaluation
    verbose = 0
    if verbosity > 0:
        print 'Evaluation begins'
        verbose = 1
    outputs_pred = model.predict(X, batch_size=batch_size, verbose=verbose)
    if verbosity > 0:
        print 'Evaluation finished'

    return outputs_pred, (Y_labels, Y_offsets)


if __name__ == '__main__':
    p = input_parser()
    main(**vars(p.parse_args()))
