#!/usr/bin/env python3
"""Code to read the Twitter dataset stored in JSON format and to identify the
    language of the author.

This code only outputs log messages as results. The desired languange must be filtered through other meanas as, for example, the following shell script that analyses the log messages: for author_dir in $(grep ' Identified language: en ' <log filename> | cut -d ' ' -f 9 ); do cp -r author_dir <destination directory>; done

"""

import argparse
import logging
import glob
import os
import sys
import json
sys.path.append(os.sep.join(['..', 'json']))
import tagging_irrelevant_data
import langid


def command_line_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', '-a',
                        dest='dataset_directory',
                        required=True,
                        help='Directory containing the Twitter dataset.')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


def remove_irrelevant_information(text):

    tagged = tagging_irrelevant_data.tag_url(text, u'')
    tagged = tagging_irrelevant_data.tag_userref(tagged, u'')
    tagged = tagging_irrelevant_data.tag_hashtag(tagged, u'')
    tagged = tagging_irrelevant_data.tag_date(tagged, u'')
    tagged = tagging_irrelevant_data.tag_time(tagged, u'')
    tagged = tagging_irrelevant_data.tag_number(tagged, u'')
    return tagged


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting dataset filtering ...',
                           '\n\tdataset directory = ', args.dataset_directory,
                           '\n\tdebug = ', str(args.debug),
                         ]))

    users = glob.glob(os.sep.join([args.dataset_directory, '[0-9]*']))
    logging.info('Processing files ...')
    for user in users:
        with open(os.sep.join([user, 'tweets.json']), mode='rt', encoding='ascii') as fd:
            tweets = []
            for message in json.load(fd):
                tweets.append(message['text'])
        language = langid.classify(remove_irrelevant_information('\n'.join(tweets)))
        logging.info('User: %s - Identified language: %s - Confidence Score: %f' % (os.path.basename(user), language[0], language[1]))

    logging.info('Finished.')
