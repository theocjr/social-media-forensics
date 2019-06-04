#!/bin/bash -x

# Script for automated Theano/Lasagne installation. Worked with CUDA 9.0 (through Docker image nvidia/cuda:9.0-cudnn7-devel) and cuDNN 7.1.4.
# References:
#   http://deeplearning.net/software/theano/install_ubuntu.html
#   https://conda.io/miniconda.html
#   https://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version

function notify_error {
    echo "ERROR EXECUTING COMMAND"
    cd $INITIAL_DIR
    exit 1
}


trap notify_error ERR

INITIAL_DIR=$(pwd)

echo "##### Updating repository and upgrading libraries ..."
sudo apt-get update
sudo apt-get --yes upgrade

echo "##### Installing iPython ..."
# This first invocation of conda binary will break some times due to a bug in conda permission checking. Just repeat this invocation until works (https://github.com/conda/conda/issues/7267).
conda install --yes ipython

echo "##### Installing Theano and its pre-requisites ..."
conda install --yes cython
conda install --yes numpy
conda install --yes scipy
conda install --yes mkl mkl-service
pip install --user git+https://github.com/Theano/Theano.git#egg=Theano
conda install --yes --channel mila-udem pygpu

echo "##### Setting Theano configuration file ..."
cat << EOF >> ~/.theanorc
[cuda]
root=/usr/local/cuda

[global]
device=cuda
floatX=float32
EOF

echo "##### Installing Lasagne ..."
pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

echo "##### Cleaning up ..."
cd $INITIAL_DIR
