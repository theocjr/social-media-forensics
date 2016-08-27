#!/usr/bin/env bash

DATE=$(date '+%Y%m%d%H%M')
ROOT_DIR="/work/atheophilo/microblog_authorship_attribution"
SOURCE_DIR="${ROOT_DIR}/dataset/datasetPOS_retweets_filtered_4_words_english_tagged_thiagos_approach_ngrams"
MINIMAL_NUMBER_TWEETS="1000"
NUM_AUTHORS="50"
TWEETS_PER_USER_LIST="1 2 5"
FEATURES="word-2-gram"

for TWEETS_PER_USER in $TWEETS_PER_USER_LIST;
do
    echo " $TWEETS_PER_USER of $TWEETS_PER_USER_LIST"
    RUN_ID="pmsvm_classifier_no_cross_${MINIMAL_NUMBER_TWEETS}_${NUM_AUTHORS}_${TWEETS_PER_USER}_${FEATURES}_${DATE}"
    /work/atheophilo/microblog_authorship_attribution/code/classification/pmsvm_classifier_no_cross.py \
        --source-dir  $SOURCE_DIR \
        --output-dir "${ROOT_DIR}/experiments/${RUN_ID}" \
        --minimal-number-tweets $MINIMAL_NUMBER_TWEETS \
        --number-authors $NUM_AUTHORS \
        --number-tweets $TWEETS_PER_USER \
        --features $FEATURES \
        --debug > "${ROOT_DIR}/experiments/${RUN_ID}.log" 2>&1
done

