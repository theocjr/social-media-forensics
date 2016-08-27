#!/usr/bin/env bash

ROOT_DIR="/work/atheophilo/microblog_authorship_attribution"
EXPERIMENT_DIR="${ROOT_DIR}/experiments/pmsvm_classifier_fake_1000_10_10_1000_1000_all_201605150533"
PMSVM_BIN="${ROOT_DIR}/code/PmSVM/pmsvm"
RUNS="001"
FOLDS="01 02 03 04 05 06 07 08 09 10"

for RUN in $RUNS;
do
    for FOLD in $FOLDS;
    do
        CURRENT_DIR="${EXPERIMENT_DIR}/run_${RUN}/fold_${FOLD}"
        echo "Running PmSVM on $CURRENT_DIR ..."
        echo "Init time: $(date)"
        $PMSVM_BIN ${CURRENT_DIR}/pmsvm_train.dat ${CURRENT_DIR}/pmsvm_test.dat > ${CURRENT_DIR}/pmsvm_stdout.log 3> ${CURRENT_DIR}/pmsvm_stderr.log
        grep -E '^Average accuracy = ([0-9\.]+)$' ${CURRENT_DIR}/pmsvm_stdout.log
        echo "End time: $(date)"
    done
done
