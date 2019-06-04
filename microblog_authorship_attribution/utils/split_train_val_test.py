#!/usr/bin/env python3


"""
"""


import argparse
import logging
import sys
import os
import glob
import json
import math


def command_line_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir-data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the tweets\' files are stored.')
    parser.add_argument('--prefix-dest-dir', '-b',
                        dest='prefix_dest_dir',
                        required=True,
                        help='Directory prefix where the output files will be written.')
    parser.add_argument('--train-ratio', '-t',
                        dest='train_ratio',
                        type=float,
                        default=0.7,
                        help='Ratio of samples to be used for training. Default = 0.7 .')
    parser.add_argument('--validation-ratio', '-v',
                        dest='valid_ratio',
                        type=float,
                        default=0.2,
                        help='Ratio of samples to be used for validation. Default = 0.2 .')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting splitting data ...',
                           '\n\tsource directory data = ', str(args.source_dir_data),
                           '\n\tprefix destination directory = ', str(args.prefix_dest_dir),
                           '\n\ttraining data ratio = ', str(args.train_ratio),
                           '\n\tvalidation data ratio = ', str(args.valid_ratio),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    if args.train_ratio + args.valid_ratio > 1.0:
        logging.error('Invalid training/validation/test setup. Quitting ...')
        sys.exit(1)

    logging.info('Creating output directories ...')
    dest_dir = ''.join([args.prefix_dest_dir,
                         '_train-',
                         str(args.train_ratio),
                        '_valid-',
                         str(args.valid_ratio),
                        ])
    if os.path.exists(dest_dir):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)

    logging.info('Splitting messages ...')
    for author_dir in glob.glob(os.sep.join([args.source_dir_data, '[0-9]*'])):
        with open(os.sep.join([author_dir, 'tweets.json']), mode='rt', encoding='ascii') as fd:
            messages = json.load(fd)
        train_idx = math.floor(len(messages)*args.train_ratio)
        valid_idx = train_idx + math.floor(len(messages)*args.valid_ratio)
        author_output_dir = os.sep.join([dest_dir, os.path.basename(author_dir)])
        os.makedirs(author_output_dir)
        with open(os.sep.join([author_output_dir, 'training.json']), mode='xt', encoding='ascii') as fd:
            json.dump(messages[:train_idx], fd, sort_keys=True, ensure_ascii=True)
        with open(os.sep.join([author_output_dir, 'valid.json']), mode='xt', encoding='ascii') as fd:
            json.dump(messages[train_idx:valid_idx], fd, sort_keys=True, ensure_ascii=True)
        with open(os.sep.join([author_output_dir, 'test.json']), mode='xt', encoding='ascii') as fd:
            json.dump(messages[valid_idx:], fd, sort_keys=True, ensure_ascii=True)

    logging.info('Finished.')
