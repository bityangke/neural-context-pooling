#!/bin/bash
set -e

# Create conda environment
# Generated via: $conda create --name neural-context-pooling python=2.7 numpy scipy h5py pyyaml
# Heavy versions add other packages on top (pandas, jupyter, ipdb, :smile:)
if hash conda 2>/dev/null; then
  if [ ! "$(conda env list | grep "neural-context-pooling")" ]; then
    conda env create -f environment.yml
  fi
else
  echo "Conda is not installed"
  return -1
fi

source activate neural-context-pooling
# Install Theano and Keras
pip install --no-deps git+git://github.com/Theano/Theano.git
pip install --no-deps git+git://github.com/fchollet/keras.git
