#!/usr/bin/env python3


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
import json
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
                        help='Filter out retweets (tweets marked as retweets or that contains the RT word as a meta-tag followed by a user reference). Default = False.')
    parser.add_argument('--minimal-number-words', '-m',
                        dest='min_words',
                        type=int,
                        default=3,
                        help='Minimal number of words to keep the tweets. Default = 3 .')
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
    author_dirs = glob.glob(os.sep.join([args.source_dir_data, '[0-9]*']))
    num_files = len(author_dirs)            # processing feedback
    i = 0                                   # processing feedback
    num_total_messages = 0
    num_filtered_messages = 0
    retweets_regex_mask = u'RT\s*@[0-9a-zA-Z_]+'
    for author_dir in author_dirs:
        author_id = os.path.basename(author_dir)
        sys.stdout.write(''.join(['\t', str(i), '/', str(num_files), ' files processed\r']))
        i += 1
        logging.debug(' '.join(['\tProcessing author', author_id, '...']))
        with open(os.sep.join([author_dir, 'tweets.json']), mode='rt', encoding='ascii') as fd:
            messages = json.load(fd)
        messages_filtered = []
        for message in messages:
            keep = True
            text = message['full_text'] if 'full_text' in message else message['text']  # long tweets according https://developer.twitter.com/en/docs/tweets/tweet-updates.html
            if args.filter_retweets:
                if 'retweeted_status' in message or re.search(retweets_regex_mask, text):
                    keep = False
            if len(text.split()) < args.min_words:
                keep = False
            if keep:
                messages_filtered.append({'id':message['id'], 'text':text})
            else:
                logging.debug('\t\tFiltering tweet: ' + text)

        author_dest_dir = os.sep.join([args.dest_dir, author_id])
        os.makedirs(author_dest_dir)
        with open(os.sep.join([author_dest_dir, 'tweets.json']), mode='wt', encoding='ascii') as fd:
            json.dump(messages_filtered, fd, sort_keys=True, ensure_ascii=True)
        num_total_messages += len(messages)
        num_filtered_messages += len(messages_filtered)
        logging.debug('\t{}/{} tweets of author {} were filtered out.'.format(len(messages_filtered), len(messages), author_id))

    logging.info('{}/{} messages filtered out.'.format(num_filtered_messages, num_total_messages))
    logging.info('Finished.')
