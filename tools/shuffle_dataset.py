import os
from argparse import ArgumentParser

import h5py
import numpy as np

from ncp.preprocessing import held_out

HDF5_DATASETS = ['Representation', 'Labels', 'Targets', 'RegressionMask']


def offline_preproc_activitynet(filename, newfile, hdf5_datasets=HDF5_DATASETS,
                                ratio=0.45):
    """Offline preprocessing of ActivityNet HDF5 file
       It performs several cleaning and pre-processing tasks like:
           1. discard entries (rows) with NaN based of feature representation
           2. Shuffle dataset to break inertia during saving process
           3. Allocate samples for validation set at the end of the file

    Parameters
    ----------
    filename : str
        Path hdf5 file
    newfile : str
        Path hdf5 file already preprocess
    hdf5_datasets : list
        List of string with names of dataset (HDF5 context) to pre-process

    """
    if not os.path.isfile(filename):
        raise IOError('Unexistent file {}'.format(filename))

    with h5py.File(filename) as fid:
        dataset_tensors = [fid[i][...] for i in hdf5_datasets]

    # Remove nan if any
    representation = dataset_tensors[0]
    idx_rm = np.isnan(representation).sum(axis=1) > 1
    if idx_rm.sum() > 1:
        print 'KaBOOM!!! Houston: dataset contains NaN values'
        dataset_tensors = (i[~idx_rm, ...] for i in dataset_tensors)

    # Caveat: assume hdf5_datasets have desired order, sorry!
    labels = dataset_tensors[1]
    idx = held_out(labels, dataset_tensors, ratio)

    with h5py.File(newfile, 'w') as fid:
        for i, tensor in enumerate(dataset_tensors):
            if tensor.ndim > 1:
                chunks = (1,) + tensor.shape[1:]
            else:
                chunks = True

            fid.create_dataset(
                hdf5_datasets[i], data=tensor[idx, ...], chunks=chunks,
                compression="gzip", compression_opts=9)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-if', '--filename', default='', help='HDF5 provided')
    p.add_argument('-of', '--newfile', default='', help='New HDF5')
    p.add_argument('-r', '--ratio', default=0.45, type=float,
                   help='Ratio of videos for validation')
    offline_preproc_activitynet(**vars(p.parse_args()))
