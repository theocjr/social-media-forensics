#!/usr/bin/env python


"""
Program to trigger a Random Forest (RF) classifier based on scikit-learn code. Its input are the n-grams output by ngrams_generator.py code. Besides doing the classification, it also calculates the importance of each feature used.

The feature importance rational is: for each RF model, the 100 most important discriminative n-grams are chosen. They are accounted by their rank in the importance order (the bigger the better). For each n-gram, it is mapped to its feature type (char-4-gram, word-1-gram, ...) and a counter for each feature type is incremented with its rank. At the end a normalization is done and each feature has a number between 0 and 1 representing its relative importance.

Logic in pseudo-code
    1 - Filter out author with few tweets (through the prefix at filename)
    2 - For each run
        2.1 - Sample the authors
        2.2 - For each sampled author
            2.2.1 - Read and sample the tweets ngrams
        2.3 - Fold the sampled dataset (list of histograms)
        2.4 - For each fold
            2.4.1 - Remove 'hapax legomena' from the training set
            2.4.2 - Fit the vectorizer based on training set
            2.4.3 - Define training/test feature vectors through the vectorizer learned
            2.4.4 - Train and run the classifier
            2.4.5 - Account for the feature importance
            2.4.6 - Save feature importance data
            2.4.7 - Register accuracy for this fold
        2.5 - Calculate accuracy for this run
    3 - Calculate accuracy for this experiment
    4 - Calculate feature importance for this experiment
"""


import argparse
import logging
import os
import sys
import glob
import random
import sklearn.cross_validation
import sklearn.feature_extraction
import sklearn.externals.joblib
import sklearn.ensemble
import copy
import scipy
import numpy


features_list = ['char-4-gram',
                 'word-1-gram',
                 'word-2-gram',
                 'word-3-gram',
                 'word-4-gram',
                 'word-5-gram',
                 'pos-1-gram',
                 'pos-2-gram',
                 'pos-3-gram',
                 'pos-4-gram',
                 'pos-5-gram',
                ]


def command_line_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir-data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the tweets\' files are stored.')
    parser.add_argument('--output-dir', '-b',
                        dest='output_dir',
                        required=True,
                        help='Directory where the output files will be written.')
    parser.add_argument('--minimal-number-tweets', '-t',
                        dest='min_tweets',
                        type=int,
                        default=1000,
                        help='Minimal number of tweets an author must have. Default = 1000.')
    parser.add_argument('--validation-folding', '-v',
                        dest='validation_folding',
                        type=int,
                        default=10,
                        help='Number of cross-validation folds. Default = 10.')
    parser.add_argument('--repetitions', '-r',
                        dest='repetitions',
                        type=int,
                        default=10,
                        help='Number of repetitions for the experiment. Default = 10.')
    parser.add_argument('--number-authors', '-u',
                        dest='num_authors',
                        type=int,
                        default=50,
                        help='Number of authors. Default = 50.')
    parser.add_argument('--number-tweets', '-w',
                        dest='num_tweets',
                        type=int,
                        default=500,
                        help='Number of tweets per author. Default = 500.')
    parser.add_argument('--features', '-f',
                        choices = ['all'] + features_list,
                        nargs = '+',
                        default=['all'],
                        help='Features to be used in classification. Default = all.')
    parser.add_argument('--number-trees', '-e',
                        dest='num_trees',
                        type=int,
                        default=500,
                        help='Number of trees in the Random Forest classifier. Default = 500.')
    parser.add_argument('--number-most-important-features', '-m',
                        dest='num_most_important_features',
                        type=int,
                        default=100,
                        help='Number of most important features of a RF classifier that will be analysed. Default = 100.')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


def filter_authors(source_dir_data, threshold):
    selected_filenames = []
    for filename in glob.glob(os.sep.join([source_dir_data, '*'])):
        if threshold <= int(os.path.basename(filename).split('_')[0]):
            selected_filenames.append(filename)
    return selected_filenames


def sample_tweets(authors_list, num_tweets, features):
    tweets_sampled = {}
    feature_kind_dict = {}      # Maps each gram (feature) to its kind (gram-4-gram, word-1-gram, pos-1-gram, ...)
    for author in authors_list:
        logging.debug(''.join(['\t\tSampling tweets from author ', os.path.basename(author), ' ...']))
        author_features_list = []
        for feature in features:
            logging.debug(''.join(['\t\t\tReading feature ' , feature, ' ...']))
            author_features_list.append(sklearn.externals.joblib.load(''.join([author, os.sep, feature, '.pkl'])))
            for tweet_histogram in author_features_list[-1]:        # mount the feature_kind_dict mapping
                for gram in tweet_histogram.keys():
                    feature_kind_dict[gram] = feature

        if len(author_features_list) > 0:
            tweets_sampled[author] = list(author_features_list[0])
        for feature in author_features_list[1:]:
            for i in range(len(tweets_sampled[author])):
                tweets_sampled[author][i].update(feature[i])

        random.shuffle(tweets_sampled[author])
        tweets_sampled[author] = tweets_sampled[author][0:num_tweets]

    return tweets_sampled, feature_kind_dict


def build_inverse_vocabulary_array(vocabulary):
    inverse_vocabulary_array = numpy.empty(len(vocabulary.keys()), dtype='object')
    for key in vocabulary.keys():
        inverse_vocabulary_array[vocabulary[key]] = key
    return inverse_vocabulary_array


def remove_hapax_legomena(histograms_list):
    if not histograms_list:
        return

    # sum the occurrence of each feature (gram) through numpy operations
    vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
    feature_occurrence_sum = numpy.sum(vectorizer.fit_transform(histograms_list), axis=0)

    # build an array where the elements are the features (grams) in the same order of the feature_occurence_sum columns
    inverse_vocabulary_array = build_inverse_vocabulary_array(vectorizer.vocabulary_)

    # find the hapax legomena
    hapax_legomena = []
    for i in range(len(feature_occurrence_sum)):
        if feature_occurrence_sum[i] == 1.0:
            hapax_legomena.append(inverse_vocabulary_array[i])

    # remove the hapax legomena
    for hapax in hapax_legomena:
        for histogram in histograms_list:
            if hapax in histogram:
                del histogram[hapax]


def fit_classify(num_trees, x_train, y_train, x_test, y_test):
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators = num_trees, n_jobs = 6)
    rf.fit(x_train, y_train)

    # return the fitted model and the accuracy in percentage
    return (rf, rf.score(x_test, y_test) * 100.0)


if  __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()
    if 'all' in args.features:
        args.features = features_list

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting the classification ...',
                           '\n\tsource directory data = ', args.source_dir_data,
                           '\n\toutput directory = ', args.output_dir,
                           '\n\tminimal number of tweets = ', str(args.min_tweets),
                           '\n\tnumber of folds in cross validation = ', str(args.validation_folding),
                           '\n\tnumber of repetitions = ', str(args.repetitions),
                           '\n\tnumber of authors = ', str(args.num_authors),
                           '\n\tnumber of tweets = ', str(args.num_tweets),
                           '\n\tfeatures = ', str(args.features),
                           '\n\tnumber of trees = ', str(args.num_trees),
                           '\n\tnumber of most important features = ', str(args.num_most_important_features),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    if args.num_tweets % args.validation_folding != 0:
        logging.error('Number of tweets per author must be multiple of validation folding value. Quitting ...')
        sys.exit(1)

    logging.info('Creating output directory ...')
    if os.path.exists(args.output_dir):
        logging.error('Output directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.output_dir)

    logging.info(''.join(['Filtering out authors with less than ', str(args.min_tweets), ' tweets ...']))
    authors_list = filter_authors(args.source_dir_data, args.min_tweets)
    if len(authors_list) < args.num_authors:
        logging.error('Too few author\'s filenames to sample. Exiting ...')
        sys.exit(1)
    logging.info(''.join(['Selected ', str(len(authors_list)), ' authors for the experiment.']))
    with open(''.join([args.output_dir, os.sep, 'filtered_authors.txt']), mode='w') as fd:
        fd.write('\n'.join(authors_list))
    
    accuracy_accumulator = 0.0

    # the block below are the variables used to account for the final feature kind importance
    # dictionary indexed by the feature kind (char-4-gram, word-1-gram, ...) that contains the sum of the ranks of features of that kind
    feature_kind_importance_accumulator = {}
    for feature in args.features:
        feature_kind_importance_accumulator[feature] = 0
    # the normalization is the sum of all the ranks from 1 to args.num_most_important_features (arithmetic progression sum) times the number of folds times the number of runs
    feature_importance_normalization = (1.0 + args.num_most_important_features) * (args.num_most_important_features / 2.0) * args.validation_folding * args.repetitions

    for run in range(1, args.repetitions+1):
        logging.info('Run ' + str(run))
        run_dir = ''.join([args.output_dir, os.sep,'run_', str(run).zfill(3)])
        os.makedirs(run_dir)
    
        logging.info('\tSampling the authors ...')
        authors_sampled = list(authors_list) # copy the list
        random.shuffle(authors_sampled)
        authors_sampled = authors_sampled[0:args.num_authors]
        with open(''.join([run_dir, os.sep, 'sampled_authors.txt']), mode='w') as fd:
            fd.write('\n'.join(authors_sampled))

        logging.info('\tSampling the tweets ...')
        tweets_sampled, feature_kind_dict = sample_tweets(authors_sampled, args.num_tweets, args.features)

        fold_accuracy_accumulator = 0.0
        logging.debug('\tFolding the dataset ...')
        folds = sklearn.cross_validation.KFold(args.num_tweets, n_folds=args.validation_folding)
        fold_id = 0
        for train, test in folds:
            fold_id += 1
            logging.info(''.join(['\tRunning fold ', str(fold_id), ' ...']))
            fold_dir = ''.join([run_dir, os.sep,'fold_', str(fold_id).zfill(2)])
            os.makedirs(fold_dir)
            train_list = []
            test_list = []
            logging.debug('\t\tBuilding training/test sets...')
            fold_tweets_sampled = copy.deepcopy(tweets_sampled)     # need to copy recursively all the data so that the hapax legomena step do not interfere in the other folds
            author_train_start_idx = 0
            class_id = 0
            y_train = []
            y_test = []
            for author in fold_tweets_sampled.keys():
                for idx in train:
                    train_list.append(fold_tweets_sampled[author][idx])
                logging.debug(''.join(['\t\t\tRemoving \'hapax legomena\' for author ', os.path.basename(author), '...']))
                remove_hapax_legomena(train_list[author_train_start_idx:])
                author_train_start_idx += len(train)
                for idx in test:
                    test_list.append(fold_tweets_sampled[author][idx])
                y_train += [class_id] * len(train)
                y_test += [class_id] * len(test)
                class_id += 1
            logging.debug('\t\tFitting and vectorizing the training set ...')
            vectorizer = sklearn.feature_extraction.DictVectorizer()
            x_train = vectorizer.fit_transform(train_list)
            logging.debug('\t\tVectorizing the test set ...')
            x_test = vectorizer.transform(test_list)
            logging.debug('\t\tTransforming the feature vector in a binary activation feature vector ...')
            x_train = x_train.astype(bool).astype(int)
            x_test = x_test.astype(bool).astype(int)

            logging.debug('\t\tTraining and running the classifier ...')
            logging.debug(''.join(['\t\tFeature vector training size: ', str(x_train.shape)]))
            model, fold_accuracy = fit_classify(args.num_trees, x_train.todense(), y_train, x_test.todense(), y_test)
            fold_accuracy_accumulator += fold_accuracy

            logging.debug('\t\tAccounting for feature importance')
            inverse_vocabulary_array = build_inverse_vocabulary_array(vectorizer.vocabulary_)
            most_important_features_idxs = numpy.argsort(model.feature_importances_)[-args.num_most_important_features:]     # the indexes of the 100 most important features in ascending order of importance (the bigger the better)
            for i in range(len(most_important_features_idxs)):       # account the rank of the feature
                feature = inverse_vocabulary_array[most_important_features_idxs[i]]
                feature_kind_importance_accumulator[feature_kind_dict[feature]] += i+1

            logging.debug('\t\tSaving feature importance data ...')
            sklearn.externals.joblib.dump(vectorizer.vocabulary_, os.sep.join([fold_dir, 'vectorizer_vocabulary.pkl']))
            sklearn.externals.joblib.dump(model.feature_importances_, os.sep.join([fold_dir, 'rf_model_feature_importances.pkl']))

            logging.info(''.join(['\t\tFold ', str(fold_id), ' accuracy: ', str(fold_accuracy)]))

        run_accuracy = fold_accuracy_accumulator/args.validation_folding
        accuracy_accumulator += run_accuracy
        logging.info(''.join(['\tRun ', str(run), ' accuracy:', str(run_accuracy)]))

    logging.info(''.join(['Final accuracy: ', str(accuracy_accumulator/args.repetitions), '%']))
    logging.info('Feature importance:')
    feature_importance_counter = 0.0
    for feature_kind in args.features:
        feature_importance_counter +=  feature_kind_importance_accumulator[feature_kind]
    for feature_kind in args.features:
        logging.info(''.join(['\tFeature ', feature_kind, ' importance: ', str(feature_kind_importance_accumulator[feature_kind]/feature_importance_normalization) ]))

    logging.info('Finishing ...')
