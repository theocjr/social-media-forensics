#!/usr/bin/env python


"""
Apart from the inclusion of new values for char grams, this code is very similar to the one in ../dataset_pre_processing directory with the difference that, in order to compare the features with the same dataset, the sampling of authors/tweets is done once in this code and not in pmsvm_classifier_char-grams.py.
"""


import argparse
import logging
import os
import sys
import glob
import random
import messages_persistence
import time
from nltk.util import ngrams
import numpy
import sklearn.feature_extraction
import sklearn.externals.joblib
import re


features_list = ['char-1-gram',
                 'char-2-gram',
                 'char-3-gram',
                 'char-4-gram',
                 'char-5-gram',
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
    parser.add_argument('--minimal-number-tweets', '-t',
                        dest='min_tweets',
                        type=int,
                        default=1000,
                        help='Minimal number of tweets an author must have. Default = 1000.')
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
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


def filter_authors(source_dir_data, threshold):
    selected_filenames = []
    for filename in glob.glob(os.sep.join([args.source_dir_data, '*.dat'])):
        if threshold <= int(os.path.basename(filename).split('_')[0]):
            selected_filenames.append(filename)
    return selected_filenames


def sample_tweets(authors_list, num_tweets, features):
    tweets_sampled = {}
    for author_filename in authors_list:
        logging.debug(''.join(['\tSampling tweets from author ', os.path.basename(author_filename), ' ...']))
        tweets = messages_persistence.read(author_filename)
        random.shuffle(tweets)
        author_id = os.path.splitext(os.path.basename(author_filename))[0]
        tweets_sampled[author_id] = tweets[0:num_tweets]
    return tweets_sampled


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


def ngrams_generator(tweets, features, save_dir):
    char_word_len = None        # variable used for checking the consistency between char/word and pos-tags grams.
    
    # char grams generation
    for i in range(1,6):
        feature = ''.join(['char-', str(i), '-gram'])
        if feature in features:
            logging.debug(''.join(['\t\tGenerating ', feature, ' features ...']))
            start_time = time.time()
            gram_list = []
            for tweet in tweets:
                grams = ngrams(u' ' + tweet['tweet'] + u' ', i)         # adding space as delimiter
                gram_list.append(grams_histogram(grams))
            if not char_word_len:
                char_word_len = len(gram_list)
            logging.debug('\t\t\tRemoving \'hapax legomena\' ...')
            remove_hapax_legomena(gram_list)
            logging.debug(''.join(['\t\t\t', feature, ' generation time: ', str("%0.3f" % (time.time()-start_time)), ' seconds.']))
            sklearn.externals.joblib.dump(gram_list, ''.join([save_dir, os.sep, feature, '.pkl']))

    # word grams generation
    tweets_words = []
    logging.debug('\t\tRemoving the punctuation of tweets to generate word grams ...')
    punctuation = u'\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^\\_\\`\\{\\|\\}\\~'     # source: re.escape(string.punctuation)
    for tweet in tweets:
        words = re.sub(u''.join([u'[', punctuation, u']']), '', tweet['tweet']).split()
        tweets_words.append([u'\x02'] + words + [u'\x03'])          # apply \x02 and \x03 as begin/end identifiers
    for i in range(1,6):
        feature = ''.join(['word-', str(i), '-gram'])
        if feature in features:
            logging.debug(''.join(['\t\tGenerating ', feature, ' features ...']))
            start_time = time.time()
            gram_list = []
            for tweet in tweets_words:
                if i == 1:
                    grams = ngrams(tweet[1:-1], i)                  # do not consider begin/end identifiers in case of word-1-grams
                else:
                    grams = ngrams(tweet, i)
                gram_list.append(grams_histogram(grams))
            if not char_word_len:
                char_word_len = len(gram_list)
            logging.debug('\t\t\tRemoving \'hapax legomena\' ...')
            remove_hapax_legomena(gram_list)
            logging.debug(''.join(['\t\t\t', feature, ' generation time: ', str("%0.3f" % (time.time()-start_time)), ' seconds.']))
            sklearn.externals.joblib.dump(gram_list, ''.join([save_dir, os.sep, feature, '.pkl']))
    
    # POS-Tag grams generation
    tweets_pos = []
    for tweet in tweets:
        if tweet['pos']:
            tweets_pos.append([u'\x02'] + tweet['pos'].split() + [u'\x03'])     # apply \x02 and \x03 as begin/end identifiers
    for i in range(1,6):
        feature = ''.join(['pos-', str(i), '-gram'])
        if feature in features:
            logging.debug(''.join(['\t\tGenerating ', feature, ' features ...']))
            start_time = time.time()
            gram_list = []
            for tweet in tweets_pos:
                if i == 1:
                    grams = ngrams(tweet[1:-1], i)                  # do not consider begin/end identifiers in case of pos-1-grams
                else:
                    grams = ngrams(tweet, i)
                gram_list.append(add_postag_id(grams_histogram(grams)))         # add an element in the pos-tag gram identifier to not mix up these grams with other char/word grams
            if char_word_len and char_word_len != len(gram_list):
                logging.error(''.join(['Tweet messages and POS Tags with different sizes: ', str(char_word_len), ' and ', str(len(gram_list)), ' respectively. Quiting ...']))
                sys.exit(1)
            logging.debug('\t\t\tRemoving \'hapax legomena\' ...')
            remove_hapax_legomena(gram_list)
            logging.debug(''.join(['\t\t\t', feature, ' generation time: ', str("%0.3f" % (time.time()-start_time)), ' seconds.']))
            sklearn.externals.joblib.dump(gram_list, ''.join([save_dir, os.sep, feature, '.pkl']))


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
                           '\n\tminimal number of tweets = ', str(args.min_tweets),
                           '\n\tnumber of authors = ', str(args.num_authors),
                           '\n\tnumber of tweets = ', str(args.num_tweets),
                           '\n\tfeatures = ', str(args.features),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Creating output directory ...')
    if os.path.exists(args.dest_dir):
        logging.error('Output directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.dest_dir)

    logging.info(''.join(['Filtering out authors with less than ', str(args.min_tweets), ' tweets ...']))
    authors_list = filter_authors(args.source_dir_data, args.min_tweets)
    if len(authors_list) < args.num_authors:
        logging.error('Too few author\'s filenames to sample. Exiting ...')
        sys.exit(1)
    logging.info(''.join(['Selected ', str(len(authors_list)), ' authors for n-grams generation.']))
    with open(''.join([args.dest_dir, os.sep, 'filtered_authors.txt']), mode='w') as fd:
        fd.write('\n'.join(authors_list))

    logging.info('Sampling the authors ...')
    random.shuffle(authors_list)
    authors_list = authors_list[0:args.num_authors]
    with open(''.join([args.dest_dir, os.sep, 'sampled_authors.txt']), mode='w') as fd:
        fd.write('\n'.join(authors_list))

    logging.info('Reading dataset and sampling the tweets ...')
    tweets_sampled = sample_tweets(authors_list, args.num_tweets, args.features)

    logging.info('Generating n-grams ...')
    for author_id in tweets_sampled:
        logging.debug(''.join(['\tGenerating n-grams for author ', author_id, ' ...']))
        author_dir = os.sep.join([args.dest_dir, author_id])
        os.makedirs(author_dir)
        ngrams_generator(tweets_sampled[author_id], args.features, author_dir)

    logging.info('Finishing ...')
    
