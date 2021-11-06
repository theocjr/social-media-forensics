#!/usr/bin/env python


"""
Code for generating ngrams for the messages presented in the dataset.
The output are files in Python's built-in persistence model format implemented
    by sklearn, one file for each feature of each author (char-4-gram, 
    word-1-gram, word-2-gram, ...) to be fed to the classifiers.
"""


import argparse
import logging
import os
import sys
import glob
import messages_persistence
from nltk.util import ngrams
import numpy
import sklearn.feature_extraction
import sklearn.externals.joblib
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
    parser.add_argument('--dest-dir', '-b',
                        dest='dest_dir',
                        required=True,
                        help='Directory where the output files will be written.')
    parser.add_argument('--features', '-f',
                        choices = ['all'] + features_list,
                        nargs = '+',
                        default=['all'],
                        help='Features to be used in classification. Default = all.')
    parser.add_argument('--remove_hapax_legomena', '-r',
                        dest='remove_hapax_legomena',
                        action='store_true',
                        default=False,
                        help='Remove hapax legomena from the generated features. Default = False.')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


def grams_histogram(grams):
    histogram = {}
    for gram in grams:
        if gram in histogram:
            histogram[gram] += 1
        else:
            histogram[gram] = 1
    return histogram


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


def add_postag_id(histogram):
    aux = {}
    postag_id = 1
    for gram in histogram.keys():
        aux[(postag_id, gram)] = histogram[gram]
    return aux


def ngrams_generator(tweets, features, save_dir, clean_hapax_legomena):
    char_word_len = None
    if 'char-4-gram' in features:
        logging.debug('\tGenerating char-4-gram features ...')
        gram_list = []
        for tweet in tweets:
            grams = ngrams(u' ' + tweet['tweet'] + u' ', 4)         # adding space as delimiter
            gram_list.append(grams_histogram(grams))

        char_word_len = len(gram_list)
        if clean_hapax_legomena:
            logging.debug('\tRemoving \'hapax legomena\' ...')
            remove_hapax_legomena(gram_list)
        sklearn.externals.joblib.dump(gram_list, os.sep.join([save_dir, 'char-4-gram.pkl']))

    tweets_words = []
    logging.debug('\tRemoving the punctuation of tweets to generate word grams ...')
    punctuation = u'\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^\\_\\`\\{\\|\\}\\~'     # source: re.escape(string.punctuation)
    for tweet in tweets:
        words = re.sub(u''.join([u'[', punctuation, u']']), '', tweet['tweet']).split()
        tweets_words.append([u'\x02'] + words + [u'\x03'])          # apply \x02 and \x03 as begin/end identifiers

    for i in range(1,6):
        if ''.join(['word-', str(i), '-gram']) in features:
            logging.debug(''.join(['\tGenerating word-', str(i), '-gram features ...']))
            gram_list = []
            for tweet in tweets_words:
                if i == 1:
                    grams = ngrams(tweet[1:-1], i)                  # do not consider begin/end identifiers in case of word-1-grams
                else:
                    grams = ngrams(tweet, i)
                gram_list.append(grams_histogram(grams))
            if not char_word_len:
                char_word_len = len(gram_list)
            if clean_hapax_legomena:
                logging.debug('\tRemoving \'hapax legomena\' ...')
                remove_hapax_legomena(gram_list)
            sklearn.externals.joblib.dump(gram_list, ''.join([save_dir, os.sep, 'word-', str(i), '-gram.pkl']))

    tweets_pos = []
    for tweet in tweets:
        if tweet['pos']:
            tweets_pos.append([u'\x02'] + tweet['pos'].split() + [u'\x03'])     # apply \x02 and \x03 as begin/end identifiers
    for i in range(1,6):
        if ''.join(['pos-', str(i), '-gram']) in features:
            logging.debug(''.join(['\tGenerating pos-', str(i), '-gram features ...']))
            gram_list = []
            for tweet in tweets_pos:
                if i == 1:
                    grams = ngrams(tweet[1:-1], i)                  # do not consider begin/end identifiers in case of pos-1-grams
                else:
                    grams = ngrams(tweet, i)
                gram_list.append(add_postag_id(grams_histogram(grams)))         # add an element in the pos-tag gram identifier to not mix up these grams with other char/word grams
            if char_word_len and char_word_len != len(gram_list):
                logging.error(''.join(['Tweet messages and POS Tags with different sizes for author ', os.path.basename(save_dir), ': ', str(char_word_len), ' and ', str(len(gram_list)), ' respectively. Quitting ...']))
                sys.exit(1)
            if clean_hapax_legomena:
                logging.debug('\tRemoving \'hapax legomena\' ...')
                remove_hapax_legomena(gram_list)
            sklearn.externals.joblib.dump(gram_list, ''.join([save_dir, os.sep, 'pos-', str(i), '-gram.pkl']))


if  __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()
    if 'all' in args.features:
        args.features = features_list

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting generating n-grams ...',
                           '\n\tsource directory data = ', args.source_dir_data,
                           '\n\toutput directory = ', args.dest_dir,
                           '\n\tfeatures = ', str(args.features),
                           '\n\tremove hapax legomena = ', str(args.remove_hapax_legomena),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Creating output directory ...')
    if os.path.exists(args.dest_dir):
        logging.error('Output directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.dest_dir)

    author_dirnames = glob.glob(os.sep.join([args.source_dir_data, '*.dat']))
    num_files = len(author_dirnames)    # processing feedback
    i = 0                               # processing feedback
    logging.info('Reading dataset and generating n-grams ...')
    for filename in author_dirnames:
        sys.stderr.write(''.join(['\t', str(i), '/', str(num_files), ' files processed\r']))   # processing feedback
        i += 1
        logging.debug(''.join(['Reading tweets and generating n-grams for file ', filename, ' ...']))
        author_dir = os.sep.join([args.dest_dir, os.path.splitext(os.path.basename(filename))[0]])
        os.makedirs(author_dir)
        ngrams_generator(messages_persistence.read(filename), args.features, author_dir, args.remove_hapax_legomena)

    logging.info('Finishing ...')

