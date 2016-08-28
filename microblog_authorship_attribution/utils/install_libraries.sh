#!/bin/bash -x

function notify_error {
    echo "ERROR EXECUTING COMMAND"
    cd $INITIAL_DIR
    exit 1
}


trap notify_error ERR


apt-get --yes install vim
apt-get --yes install man
apt-get --yes install htop

apt-get --yes install python-numpy
apt-get --yes install python-scipy
apt-get --yes install python-pip
pip install --upgrade pip
pip install --upgrade nltk
pip install --upgrade scikit-learn
pip install --upgrade ipython

