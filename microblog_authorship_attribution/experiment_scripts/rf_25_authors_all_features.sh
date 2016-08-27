#!/usr/bin/env bash

DATE=$(date '+%Y%m%d%H%M')
ROOT_DIR="/work/atheophilo/microblog_authorship_attribution"
SOURCE_DIR="${ROOT_DIR}/dataset/datasetPOS_retweets_filtered_4_words_english_tagged_thiagos_approach_ngrams"
MINIMAL_NUMBER_TWEETS="1000"
VALIDATION_FOLDING="10"
REPETITIONS="10"
NUM_AUTHORS="25"
TWEETS_PER_USER_LIST="50 100 200 500 1000"
FEATURES="all"

for TWEETS_PER_USER in $TWEETS_PER_USER_LIST;
do
    echo " $TWEETS_PER_USER of $TWEETS_PER_USER_LIST"
    RUN_ID="rf_classifier_${MINIMAL_NUMBER_TWEETS}_${VALIDATION_FOLDING}_${REPETITIONS}_${NUM_AUTHORS}_${TWEETS_PER_USER}_${FEATURES}_${DATE}"
    /work/atheophilo/microblog_authorship_attribution/code/classification/rf_classifier.py \
        --source-dir  $SOURCE_DIR \
        --output-dir "${ROOT_DIR}/experiments/${RUN_ID}" \
        --minimal-number-tweets $MINIMAL_NUMBER_TWEETS \
        --validation-folding $VALIDATION_FOLDING \
        --repetitions $REPETITIONS \
        --number-authors $NUM_AUTHORS \
        --number-tweets $TWEETS_PER_USER \
        --features $FEATURES \
        --debug > "${ROOT_DIR}/experiments/${RUN_ID}.log" 2>&1
done

