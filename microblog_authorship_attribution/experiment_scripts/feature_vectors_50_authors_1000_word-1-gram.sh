#!/usr/bin/env bash

DATE=$(date '+%Y%m%d%H%M')
ROOT_DIR="/work/atheophilo/microblog_authorship_attribution"
SOURCE_DIR="${ROOT_DIR}/dataset/datasetPOS_retweets_filtered_4_words_english_tagged_thiagos_approach_ngrams"
MINIMAL_NUMBER_TWEETS="1000"
VALIDATION_FOLDING="10"
REPETITIONS="10"
NUM_AUTHORS="50"
TWEETS_PER_USER="1000"
FEATURES="word-1-gram"

RUN_ID="${ROOT_DIR}/experiments/feature_vectors_${MINIMAL_NUMBER_TWEETS}_${VALIDATION_FOLDING}_${REPETITIONS}_${NUM_AUTHORS}_${TWEETS_PER_USER}_${FEATURES}_${DATE}"
/work/atheophilo/microblog_authorship_attribution/code/classification/feature_vectors_generator.py \
    --source-dir  $SOURCE_DIR \
    --output-dir ${RUN_ID} \
    --minimal-number-tweets $MINIMAL_NUMBER_TWEETS \
    --validation-folding $VALIDATION_FOLDING \
    --repetitions $REPETITIONS \
    --number-authors $NUM_AUTHORS \
    --number-tweets $TWEETS_PER_USER \
    --features $FEATURES \
    --debug > "${RUN_ID}.log" 2>&1

