#!/usr/bin/env python


"""
Program to trigger a Power Mean SVM (PmSVM) classifier. Its input are the n-grams output by ngrams_generator.py code. Before performing the PmSVM classification, a PCA decomposition is conducted in order to reduce the dimensionality.

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
            2.4.4 - Reduce dimensionality through PCA
            2.4.5 - Train and run the classifier
            2.4.6 - Register accuracy for this fold
        2.5 - Calculate accuracy for this run
    3 - Calculate accuracy for this experiment
"""


import argparse
import logging
import os
import sys
import glob
import random
import sklearn.externals.joblib
import sklearn.cross_validation
import sklearn.feature_extraction
import sklearn.decomposition
import sklearn.preprocessing
import copy
import scipy
import numpy
import itertools
import re


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
    parser.add_argument('--pca-variance', '-p',
                        dest='pca_variance',
                        type=float,
                        default=1.0,
                        help='Ratio of variance to keep. Default = 1.0.')
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
    for author in authors_list:
        logging.debug(''.join(['\t\tSampling tweets from author ', os.path.basename(author), ' ...']))
        author_features_list = []
        for feature in features:
            logging.debug(''.join(['\t\t\tReading feature ' , feature, ' ...']))
            author_features_list.append(sklearn.externals.joblib.load(''.join([author, os.sep, feature, '.pkl'])))

        if len(author_features_list) > 0:
            tweets_sampled[author] = list(author_features_list[0])
        for feature in author_features_list[1:]:
            for i in range(len(tweets_sampled[author])):
                tweets_sampled[author][i].update(feature[i])

        random.shuffle(tweets_sampled[author])
        tweets_sampled[author] = tweets_sampled[author][0:num_tweets]

    return tweets_sampled


def remove_hapax_legomena(histograms_list):
    if not histograms_list:
        return

    # sum the occurrence of each feature (gram) through numpy operations
    vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
    feature_occurrence_sum = numpy.sum(vectorizer.fit_transform(histograms_list), axis=0)

   # build an array where the elements are the features (grams) in the same order of the feature_occurence_sum columns
    inverse_vocabulary_array = numpy.empty(len(vectorizer.vocabulary_.keys()), dtype='object')
    for gram in vectorizer.vocabulary_.keys():
        inverse_vocabulary_array[vectorizer.vocabulary_[gram]] = gram

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


def classify(x_train, y_train, x_test, y_test, work_dir):
    logging.debug('\t\tScaling data ...')
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    logging.debug('\t\tFormatting and saving feature vector in libsvm format ...')
    pmsvm_classifier_train_filename = os.sep.join([work_dir, 'pmsvm_train.dat'])
    pmsvm_classifier_test_filename = os.sep.join([work_dir, 'pmsvm_test.dat'])
    pmsvm_classifier_stdout_filename = os.sep.join([work_dir, 'pmsvm_stdout.log'])
    pmsvm_classifier_stderr_filename = os.sep.join([work_dir, 'pmsvm_stderr.log'])
    with open(pmsvm_classifier_train_filename, mode='w') as fd:
        for row_idx in range(x_train.shape[0]):
            sample = [ str(y_train[row_idx, 0]) ]
            for col_idx in range(x_train.shape[1]):
                sample.append(':'.join([str(col_idx+1), str(x_train[row_idx, col_idx])]))
            sample.append('\n')
            fd.write(' '.join(sample))
    with open(pmsvm_classifier_test_filename, mode='w') as fd:
        for row_idx in range(x_test.shape[0]):
            sample = [ str(y_test[row_idx, 0]) ]
            for col_idx in range(x_test.shape[1]):
                sample.append(':'.join([str(col_idx+1), str(x_test[row_idx, col_idx])]))
            sample.append('\n')
            fd.write(' '.join(sample))

    logging.debug('\t\tTraining and running the classifier ...')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ret_code = os.system(''.join([script_dir , os.sep, 'PmSVM', os.sep, 'pmsvm',        # executable    - classifier
                                  ' ', pmsvm_classifier_train_filename,                 # argument      - train data file
                                  ' ', pmsvm_classifier_test_filename,                  # argument      - test data file
                                  ' > ', pmsvm_classifier_stdout_filename,              # standard output redirection
                                  ' 2> ', pmsvm_classifier_stderr_filename              # standard error redirection
                                 ]))
    if ret_code != 0:
        logging.error(''.join(['Error running the PmSVM classifier. Error code = ', str(ret_code), ' . Exiting ...']))
        sys.exit(1)
    with open(pmsvm_classifier_stdout_filename) as fd:
        match = re.match('^Average accuracy = ([0-9\.]+)$', fd.readlines()[-1])
    if match == None:
        logging.error('Error parsing the PmSVM classifier\'s output. Exiting ...')
        sys.exit(1)
    return float(match.groups()[0])


if  __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()
    if 'all' in args.features:
        args.features = features_list

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting the Power Mean SVM classification with PCA ...',
                           '\n\tsource directory data = ', args.source_dir_data,
                           '\n\toutput directory = ', args.output_dir,
                           '\n\tminimal number of tweets = ', str(args.min_tweets),
                           '\n\tnumber of folds in cross validation = ', str(args.validation_folding),
                           '\n\tnumber of repetitions = ', str(args.repetitions),
                           '\n\tnumber of authors = ', str(args.num_authors),
                           '\n\tnumber of tweets = ', str(args.num_tweets),
                           '\n\tfeatures = ', str(args.features),
                           '\n\tratio of variance to keep = ', str(args.pca_variance),
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

    logging.info('Compiling PmSVM classifier ...')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ret_code = os.system(''.join(['g++ -O3 ',
                            script_dir, os.sep, 'PmSVM', os.sep, 'PmSVM.cpp ',
                            '-o ', script_dir, os.sep, 'PmSVM', os.sep, 'pmsvm']))
    if ret_code != 0:
        logging.error(''.join(['Error compiling the PmSVM classifier. Error code = ', str(ret_code), ' . Exiting ...']))
        sys.exit(1)

    logging.info(''.join(['Filtering out authors with less than ', str(args.min_tweets), ' tweets ...']))
    authors_list = filter_authors(args.source_dir_data, args.min_tweets)
    if len(authors_list) < args.num_authors:
        logging.error('Too few author\'s filenames to sample. Exiting ...')
        sys.exit(1)
    logging.info(''.join(['Selected ', str(len(authors_list)), ' authors for the experiment.']))
    with open(''.join([args.output_dir, os.sep, 'filtered_authors.txt']), mode='w') as fd:
        fd.write('\n'.join(authors_list))
    
    accuracy_accumulator = 0.0
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
        tweets_sampled = sample_tweets(authors_sampled, args.num_tweets, args.features)

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
            logging.debug(''.join(['\t\tTraining set dimension: ', str(x_train.shape)]))
            logging.debug('\t\tVectorizing the test set ...')
            x_test = vectorizer.transform(test_list)
    
            logging.debug(''.join(['\t\tLearning PCA decomposition ...']))
            pca = sklearn.decomposition.PCA(n_components=args.pca_variance)
            pca.fit(x_train.todense())
            logging.debug(''.join(['\t\t\tDimensionality reduction for ', str(args.pca_variance * 100.0), '% of variance. Number of components: before = ', str(x_train.shape[1]), ', after = ', str(pca.n_components_)]))
            logging.debug('\t\t\tPCA transforming of training set ...')
            x_train_pca = pca.transform(x_train.todense())
            logging.debug('\t\t\tPCA transforming of test set ...')
            x_test_pca = pca.transform(x_test.todense())
                
            y_train = numpy.asmatrix(y_train).reshape(len(y_train), 1)
            y_test = numpy.asmatrix(y_test).reshape(len(y_test), 1)

            logging.debug('\t\tClassifying ...')
            fold_accuracy = classify(x_train_pca, y_train, x_test_pca, y_test, fold_dir)
            logging.info(''.join(['\t\tFold ', str(fold_id), ' accuracy: ', str(fold_accuracy)]))
            fold_accuracy_accumulator += fold_accuracy

        run_accuracy = fold_accuracy_accumulator/args.validation_folding
        accuracy_accumulator += run_accuracy
        logging.info(''.join(['\tRun ', str(run), ' accuracy:', str(run_accuracy)]))

    logging.info(''.join(['Final accuracy: ', str(accuracy_accumulator/args.repetitions), '%']))
    logging.info('Finishing ...')
