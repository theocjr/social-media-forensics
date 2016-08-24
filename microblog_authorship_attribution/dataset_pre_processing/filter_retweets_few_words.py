#!/usr/bin/env python


"""
Code for reading authors' tweets filenames and remove retweets and tweets with
    few words.
The output filtered files are stored in a separate directory.
"""


import argparse
import logging
import os
import sys
import glob
import messages_persistence
import re


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
    parser.add_argument('--filter-retweets', '-f',
                        dest='filter_retweets',
                        action='store_true',
                        default=False,
                        help='Filter out retweets (tweets that contains the RT word as a meta-tag followed by a user reference).')
    parser.add_argument('--minimal-number-words', '-m',
                        dest='min_words',
                        type=int,
                        default=3,
                        help='Minimal number of words to keep the tweets. Default = 3')
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

    logging.info(''.join(['Starting filtering out data ...',
                           '\n\tsource directory data = ', str(args.source_dir_data),
                           '\n\tdestination directory = ', str(args.dest_dir),
                           '\n\tfilter retweets = ', str(args.filter_retweets),
                           '\n\tminimal number of words = ', str(args.min_words),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Creating output directories ...')
    if os.path.exists(args.dest_dir):
        logging.error(''.join(['Destination directory ', args.dest_dir, ' already exists. Quitting ...']))
        sys.exit(1)
    os.makedirs(args.dest_dir)

    logging.info('Processing authors\' files ...')
    filenames = glob.glob(''.join([args.source_dir_data, os.sep, '*.dat']))
    num_files = len(filenames)              # processing feedback
    i = 0                                   # processing feedback
    retweets_regex_mask = u'(^RT\s)|(?<!\S)RT\s*@[0-9a-zA-Z_]{1,}(?![0-9a-zA-Z_])'      # rationale: RT at the beginning of the message or RT followed by a user reference in the middle
    for filename in filenames:
        sys.stdout.write(''.join(['\t', str(i), '/', str(num_files), ' files processed\r']))
        i += 1
        logging.debug(''.join(['Processing ', filename, ' file ...']))
        messages = messages_persistence.read(filename)
        messages_filtered = []
        for message in messages:
            keep = True
            if args.filter_retweets and re.search(retweets_regex_mask, message['tweet']):
                keep = False
            if len(message['tweet'].split()) < args.min_words:
                keep = False
            if keep:
                messages_filtered.append(message)
            else:
                logging.debug('Filtering tweet: ' + message['tweet'])

        messages_persistence.write(messages_filtered, 'full', ''.join([args.dest_dir, os.sep, os.path.basename(filename)]))
        
    logging.info('Finishing ...')
