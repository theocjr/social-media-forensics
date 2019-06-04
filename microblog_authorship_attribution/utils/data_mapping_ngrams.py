#!/usr/bin/env python3


"""
Code for transforming tweets into sequences of numbers representing characters
    ngrams.
The original Unicode characters n-grams are translated into a index map learned
    with the training data.
"""


import argparse
import logging
import os
import sys
import glob
import json
from nltk import ngrams
import collections
import sklearn


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--source-dir-data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the tweets\' files are stored.')
    parser.add_argument('--output-directory', '-b',
                        dest='output_directory',
                        required=True,
                        help='Directory used to output the results (it is supposed that each author has its own subdirectory).')
    parser.add_argument('--ngrams-size', '-s',
                        dest='ngrams_size',
                        type=int,
                        default=4,
                        help='Size of character grams to map the text into. Default = 4 .')
    parser.add_argument('--maximum-frequency', '-x',
                        dest='maximum_frequency',
                        type=float,
                        default=1.0,
                        help='Maximum frequency (in terms of authors percentage) allowed for a character n-gram to be kept. Default = 1.0 (can be present in all the authors - 100%% ) .')
    parser.add_argument('--minimum-frequency', '-n',
                        dest='minimum_frequency',
                        type=int,
                        default=1,
                        help='Minimum frequency (in terms of absolute counts) allowed for a character n-gram to be kept. Default = 1 (no discard) .')
    parser.add_argument('--message_length', '-l',
                        dest='message_length',
                        type=int,
                        default=0,
                        help='Message length to be fixed (pad or truncate if needed). Default = 0 (keep original length with no padding or truncation).')
    parser.add_argument('--minimum-length', '-m',
                        dest='minimum_length',
                        type=int,
                        default=1,
                        help='Minimum length each message must have after n-gram mapping. Default = 1.')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    args = parser.parse_args()
    if args.maximum_frequency > 1.0:
        parser.error('--maximum-frequency value must not exceed 1.0 .')
    return args


def generate_ngrams(source_directory, ngrams_size, minimum_length):
    """ Generates ngrams for the messages.

    Returns:
        Three dictionaries (indexed by author id) for training, validation and
            test data, containing the messages as n-grams sequences.
    """

    training = {}
    valid = {}
    test = {}
    discarded_count = 0
    for author_dir in glob.glob(os.sep.join([source_directory, '[0-9]*'])):
        author_id = os.path.basename(author_dir)
        for filename, tweets_dict in zip(
                                         ['training.json', 'valid.json', 'test.json'],
                                         [training, valid, test]
                                        ):
            with open(os.sep.join([author_dir, filename]), mode='rt', encoding='ascii') as fd:
                messages = json.load(fd)
            tweets_ngrams = []
            for message in messages:
                message_ngrams = list(ngrams(message['text'], ngrams_size))
                if len(message_ngrams) < minimum_length:
                    logging.debug('\tDiscarding message of author {} with {} n-grams.'.format(author_id, len(message_ngrams)))
                    discarded_count += 1
                    continue
                tweets_ngrams.append({'id':message['id'], 'text':message_ngrams})
            tweets_dict[author_id] = tweets_ngrams
    logging.info('\t{} messages with less than {} n-grams were discarded.'.format(discarded_count, minimum_length))
    return training, valid, test


def learn_ngrams_dict(data, maximum_frequency_percent, minimum_frequency):
    """ Generates a n-gram -> integer mapping.

    Returns:
        A dictionary containing the mapping.
        A collections.Counter object containing the n-grams counting.
    """

    ngrams_authors_counter = collections.Counter()  # to track the maximum frequency
    ngrams_counter = collections.Counter()          # to track the minimum frequency
    for author_id in data:
        ngrams_set = set()
        for message in data[author_id]:
            ngrams_counter.update(message['text'])
            ngrams_set.update(message['text'])
        ngrams_authors_counter.update(list(ngrams_set))

    ngrams = sorted(ngrams_counter.most_common(), key=lambda x: x[0])   # list of n-grams 'alphabetically' ordered
    ngrams_dict = { '\0' : 0 }      # 0 is the index for the unknown element (represented by the '\0' character)
    maximum_frequency = len(data) * maximum_frequency_percent
    too_frequent = 0
    less_frequent = 0
    for ngram in ngrams:
        if ngrams_authors_counter[ngram[0]] > maximum_frequency:
            logging.debug('\t{} ngram considered too frequent. Seen in {} authors.'.format(ngram[0], ngrams_authors_counter[ngram[0]]))
            too_frequent += 1
        elif ngram[1] < minimum_frequency:
            logging.debug('\t{} ngram considered less frequent. # of occurences: {}.'.format(ngram[0], ngram[1]))
            less_frequent += 1
        else:
            ngrams_dict[ngram[0]] = len(ngrams_dict)
    logging.info('\t{} character n-grams removed due to the maximum frequency restriction.'.format(too_frequent))
    logging.info('\t{} character n-grams removed due to the minimum frequency restriction.'.format(less_frequent))
    return ngrams_dict, ngrams_counter, ngrams_authors_counter


def transform_ngrams(data, mapping, message_length, filename):
    transformed_messages = []
    for message in data:
        transformed_message = [ mapping.get(gram, mapping['\0']) for gram in message['text'] ]
        if message_length:  #   truncate or pad if message_length > 0
            if len(message['text']) > message_length:
                logging.warning(''.join(['Tweet bigger than expected (', str(message_length), '). Length = ', str(len(message['text'])), '. Truncating to ', str(message_length), ' elements.']))
                transformed_message = transformed_message[:message_length]
            else:   # proceed with padding if necessary
                transformed_message += (message_length - len(transformed_message)) * [ mapping['\0'] ]
        transformed_messages.append({'id':message['id'], 'text':transformed_message})
    with open(filename, mode='xt', encoding='ascii') as fd:
        json.dump(transformed_messages, fd, sort_keys=True, ensure_ascii=True)


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting transforming data ...',
                           '\n\tsource directory data = ', args.source_dir_data,
                           '\n\toutput directory = ', args.output_directory,
                           '\n\tn-grams size = ', str(args.ngrams_size),
                           '\n\tn-grams maximum frequency = ', str(args.maximum_frequency),
                           '\n\tn-grams minimum frequency = ', str(args.minimum_frequency),
                           '\n\tmessage length = ', str(args.message_length),
                           '\n\tminimum length = ', str(args.minimum_length),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Creating output directory ...')
    if os.path.exists(args.output_directory):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.output_directory)

    logging.info('Generating n-grams ...')
    training_ngrams, validation_ngrams, test_ngrams = generate_ngrams(args.source_dir_data, args.ngrams_size, args.minimum_length)

    logging.info('Learning the n-gram mapping ...')
    ngrams_dict, ngrams_counter, ngrams_authors_counter = learn_ngrams_dict(training_ngrams, args.maximum_frequency, args.minimum_frequency)
    logging.info(''.join(['\t', str( len(ngrams_counter) - (len(ngrams_dict)-1) ), ' character ', str(args.ngrams_size), '-grams were removed due to frequency restrictions.']))
    logging.info(''.join(['\tN-grams mapping built from training data with ', str(len(ngrams_dict)), ' character ', str(args.ngrams_size), '-grams.']))
    sklearn.externals.joblib.dump(ngrams_dict, os.sep.join([args.output_directory, 'ngrams_dict.skl']))
    sklearn.externals.joblib.dump(ngrams_counter, os.sep.join([args.output_directory, 'ngrams_counter.skl']))
    sklearn.externals.joblib.dump(ngrams_authors_counter, os.sep.join([args.output_directory, 'ngrams_authors_counter.skl']))

    logging.info('Transforming messages into n-grams indexes ...')
    for author_id in training_ngrams:
        output_author_dir = os.sep.join([args.output_directory, author_id])
        os.makedirs(output_author_dir)
        for filename, data in zip(['training.json', 'valid.json', 'test.json'],
                                  [ training_ngrams, validation_ngrams, test_ngrams ],
                                 ):
            transform_ngrams(data[author_id], ngrams_dict, args.message_length, os.sep.join([output_author_dir, filename]))

    logging.info('Finished.')

