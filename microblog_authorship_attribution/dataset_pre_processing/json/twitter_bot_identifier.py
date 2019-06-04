#!/usr/bin/env python3


"""Code to identify Twitter bots by comparing the similarities of the tweets of a user.

This code only outputs log messages as results. The desired authors must be filtered through other means as, for example, a shell script that analyses the log messages.
"""


import sys
import argparse
import logging
import os
import glob
import json
import random
import math
import numpy
import difflib


def command_line_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir-data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the tweets\' files are stored.')
    parser.add_argument('--similarity_threshold', '-s',
                        dest='similarity_threshold',
                        type=float,
                        default=0.6,
                        help='Threshold of similarity mean of all the tweets of a user to be considered a bot. Default = 0.6 .')
    parser.add_argument('--slow-ratio', '-l',
                        dest='slow_ratio',
                        action='store_true',
                        default=False,
                        help='Whether to use the regular similarity ratio function (slower). Default = False.')
    parser.add_argument('--sampling-ratio', '-m',
                        dest='sampling_ratio',
                        type=float,
                        default=1.0,
                        help='Ratio of tweets sampling to perform the similarity test if the author has more than 100 tweets. Default = 1.0 (no sampling).')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


def sample(messages, sampling_ratio):
    random.shuffle(messages)
    return messages[0: math.ceil(len(messages)*sampling_ratio)]


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting Twitter bot identifier ...',
                           '\n\tsource directory data = ', str(args.source_dir_data),
                           '\n\tsimilarity threshold = ', str(args.similarity_threshold),
                           '\n\tslow ratio = ', str(args.debug),
                           '\n\tsampling ratio= ', str(args.sampling_ratio),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Processing authors\' files ...')
    users_dirs = glob.glob(os.sep.join([args.source_dir_data, '[0-9]*']))
    num_users = len(users_dirs)
    i = 1                                   # processing feedback
    for user_dir in users_dirs:
        sys.stdout.write(''.join(['Processing author ', str(i), '/', str(num_users), ' ...\r']))
        i += 1
        logging.debug(''.join(['Processing user ', os.path.basename(user_dir),  ' ...']))
        with open(os.sep.join([user_dir, 'tweets.json']), mode='rt', encoding='ascii') as fd:
            messages = json.load(fd)
        if len(messages) < 2:
            logging.warning(''.join([os.path.basename(user_dir), ' author has less than 2 tweets. Skipping ...']))
            continue
        if len(messages) > 100:
            messages = sample(messages, args.sampling_ratio)
            if len(messages) < 2:
                logging.warning(''.join(['After sampling, ' , os.path.basename(user_dir), ' author set has less than 2 tweets. Skipping ...']))
                continue
        similarity_matrix = numpy.identity(len(messages))
        for a in range(len(messages)):
            for b in range(a+1, len(messages)):
                if args.slow_ratio:
                    similarity_matrix[a,b] = similarity_matrix[b,a] = difflib.SequenceMatcher(None, messages[a]['text'], messages[b]['text']).ratio()
                else:
                    similarity_matrix[a,b] = similarity_matrix[b,a] = difflib.SequenceMatcher(None, messages[a]['text'], messages[b]['text']).quick_ratio()
        mean = (similarity_matrix.sum() - len(messages)) / (len(messages)**2 - len(messages))       # remove the diagonal from the mean
#        logging.debug(''.join([os.path.basename(user_dir), ' author similarity mean: ', str(mean), ' . Tweets similarity matrix:']))
        logging.debug(''.join([os.path.basename(user_dir), ' author similarity mean: ', str(mean),]))
#        if args.debug:
#            print(similarity_matrix)
        if mean > args.similarity_threshold:
            logging.info(''.join([os.path.basename(user_dir), ' author identified as a bot. Similarity mean: ', str(mean)]))

    logging.info('Finishing ...')
