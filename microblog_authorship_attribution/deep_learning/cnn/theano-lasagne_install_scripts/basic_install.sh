#!/bin/bash -x

function log {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] - LOGGER - $@"
}

function notify_error {
    echo "ERROR EXECUTING COMMAND"
    cd $INITIAL_DIR
    exit 1
}


trap notify_error ERR

log "Creating user $OUTSIDE_USER and basic config files ..."
groupadd --gid $OUTSIDE_GID $OUTSIDE_GROUP
useradd --create-home --uid $OUTSIDE_UID --gid $OUTSIDE_GID $OUTSIDE_USER --shell /bin/bash
cp ../../../utils/configs/.bashrc /home/${OUTSIDE_USER}/
chown -R ${OUTSIDE_USER}:${OUTSIDE_GROUP} /home/${OUTSIDE_USER}/.bashrc
touch /home/${OUTSIDE_USER}/.bash_history
chown -R ${OUTSIDE_USER}:${OUTSIDE_GROUP} /home/${OUTSIDE_USER}/.bash_history
cp -r ../../../utils/configs/.vim* /home/${OUTSIDE_USER}/
chown -R ${OUTSIDE_USER}:${OUTSIDE_GROUP} /home/${OUTSIDE_USER}/.vim*
chown -R ${OUTSIDE_USER}:${OUTSIDE_GROUP}  /home/${OUTSIDE_USER}/

log "Updating and upgrading APT packages ..."
apt-get update
apt-get --yes upgrade

log "Installing and configuring sudo ..."
apt-get --yes install sudo
usermod -aG sudo ${OUTSIDE_USER}
sed --in-place=.original "\$ a${OUTSIDE_USER} ALL=(ALL) NOPASSWD: ALL" /etc/sudoers

log "Installing basic utility packages ..."
apt-get --yes install vim
apt-get --yes install man
apt-get --yes install htop

log "Installing and configuring Git ..."
apt-get --yes install git
git config --global user.email "antonio.theophilo@gmail.com"
git config --global user.name "Antonio Theophilo"
git config --global core.editor "vim -c 'set spell'"

log "Installing Python and machine learning basic libraries ..."
apt-get --yes install python
apt-get --yes install python-pip
pip install --upgrade pip

log "Finished"

#sudo --user=$OUTSIDE_USER --login
