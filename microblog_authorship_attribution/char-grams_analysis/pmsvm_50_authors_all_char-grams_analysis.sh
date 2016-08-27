#!/usr/bin/env bash

DATE=$(date '+%Y%m%d%H%M')
ROOT_DIR="/work/atheophilo/microblog_authorship_attribution"
SOURCE_DIR="${ROOT_DIR}/dataset/datasetPOS_retweets_filtered_4_words_english_tagged_thiagos_approach_ngrams-char-grams-analysis"
VALIDATION_FOLDING="10"
REPETITIONS="10"
FEATURES="all"

echo "Running ... "
RUN_ID="pmsvm_classifier_char-grams_analysis_${VALIDATION_FOLDING}_${REPETITIONS}_${FEATURES}_${DATE}"
/work/atheophilo/microblog_authorship_attribution/code/char-grams_analysis/pmsvm_classifier_char-grams.py \
    --source-dir  $SOURCE_DIR \
    --output-dir "${ROOT_DIR}/experiments/char-grams_analysis/${RUN_ID}" \
    --validation-folding $VALIDATION_FOLDING \
    --repetitions $REPETITIONS \
    --features $FEATURES \
    --debug > "${ROOT_DIR}/experiments/char-grams_analysis/${RUN_ID}.log" 2>&1
echo "Done!"
