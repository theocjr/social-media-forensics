#!/usr/bin/env python3


"""
Code to identify anomalous user by analising the number of followers and friends of a user.
"""


import sys
import argparse
import logging
import os
import glob
import json


def command_line_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir-data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the users\' files are stored.')
    parser.add_argument('--followers-threshold', '-o',
                        dest='followers_threshold',
                        type=int,
                        default=1000,
                        help='Number of followers to consider a user as anomalous user (celebrity or service). Default = 1000 .')
    parser.add_argument('--friends-threshold', '-r',
                        dest='friends_threshold',
                        type=int,
                        default=1000,
                        help='Number of friends to consider a user as anomalous user (bot). Default = 1000 .')
#    parser.add_argument('--followers-ratio', '-l',
#                        dest='followers_ratio',
#                        type=float,
#                        default=10.0,
#                        help='Threshold of followers to consider a user as anomalous user (celebrity or #service). The number of followers must not be greater than this threshold times the number of friends. Default = 10.0 .')
#    parser.add_argument('--friends-ratio', '-i',
#                        dest='friends_ratio',
#                        type=float,
#                        default=10.0,
#                        help='Threshold of friends to consider a user as anomalous user (bot). The number of friends must not be greater than this threshold times the number of followers. Default = 10.0 .')
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

    logging.info(''.join(['Starting anomalous user identifier ...',
                           '\n\tsource directory data = ', str(args.source_dir_data),
                           '\n\tfollowers threshold = ', str(args.followers_threshold),
                           '\n\tfriends threshold = ', str(args.friends_threshold),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Processing authors\' files ...')
    users = glob.glob(os.sep.join([args.source_dir_data, '*.json']))
    for user in users:
        with open(user, mode='rt', encoding='ascii') as fd:
            user_info = json.load(fd)
        if user_info['followers_count'] > args.followers_threshold or user_info['friends_count'] > args.friends_threshold:
            logging.info('Author {} (id {} ) identified as an anomalous user. followers count: {} ; friends count: {} ; number of messages {}'.format(user_info['screen_name'], user_info['id_str'], user_info['followers_count'], user_info['friends_count'], user_info['statuses_count']))
        else:
            logging.info('Author {} (id {} ) identified as a normal user. followers count: {} ; friends count: {} ; number of messages {}'.format(user_info['screen_name'], user_info['id_str'], user_info['followers_count'], user_info['friends_count'], user_info['statuses_count']))

    logging.info('Finishing ...')
