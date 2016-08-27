#!/usr/bin/env bash

DATE=$(date '+%Y%m%d%H%M')
NUMBER_TREES="500"
FEATURES="word-3-gram"
ROOT_DIR="/work/atheophilo/microblog_authorship_attribution"
SOURCE_DIR="${ROOT_DIR}/experiments/feature_vectors_1000_10_10_50_1000_${FEATURES}_201605162228"
RUN_ID="${ROOT_DIR}/experiments/rf_classifier_fest_1000_10_10_50_1000_${FEATURES}_${NUMBER_TREES}_${DATE}"

/work/atheophilo/microblog_authorship_attribution/code/classification/rf_classifier_fest.py \
    --source-dir  $SOURCE_DIR \
    --output-dir "${RUN_ID}" \
    --number-trees $NUMBER_TREES \
    --debug > "${RUN_ID}.log" 2>&1

