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

echo "##### Installing Miniconda ..."
MINICONDA_HOME="/home/${OUTSIDE_USER}/miniconda2"
sudo apt-get --yes install wget
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh -p ${MINICONDA_HOME} -b

CUDA_HOME="/usr/local/cuda"
echo "##### Setting environment variables ..."
cat << EOF >> ~/.bashrc

# Miniconda binaries
export PATH=${MINICONDA_HOME}/bin\${PATH:+:\${PATH}}

# CUDA environment variables
export CUDA_HOME=${CUDA_HOME}
export PATH=\${CUDA_HOME}/bin\${PATH:+:\${PATH}}
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64/:\${CUDA_HOME}/extras/CUPTI/lib64/\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOF

echo "##### Cleaning up ..."
cd $INITIAL_DIR
