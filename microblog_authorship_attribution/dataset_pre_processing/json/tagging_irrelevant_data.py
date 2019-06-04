#!/usr/bin/env python3
"""
Code for tagging irrelevant data as numbers, dates, times, URLs, hashtags and
    user references.
The output are JSON files containing only the tweets with the tagging applied.
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
    parser.add_argument('--dataset_directory', '-a',
                        dest='dataset_directory',
                        required=True,
                        help='Directory containing the Twitter dataset.')
    parser.add_argument('--destination-directory', '-b',
                        dest='destination_directory',
                        required=True,
                        help='Directory where the text files will be stored.')
    parser.add_argument('--no-number-tag', '-n',
                        dest='no_number',
                        action='store_true',
                        default=False,
                        help='Do not tag numbers.')
    parser.add_argument('--no-date-tag', '-t',
                        dest='no_date',
                        action='store_true',
                        default=False,
                        help='Do not tag dates.')
    parser.add_argument('--no-time-tag', '-i',
                        dest='no_time',
                        action='store_true',
                        default=False,
                        help='Do not tag times.')
    parser.add_argument('--no-url-tag', '-u',
                        dest='no_url',
                        action='store_true',
                        default=False,
                        help='Do not tag URLs.')
    parser.add_argument('--no-hashtag-tag', '-s',
                        dest='no_hashtag',
                        action='store_true',
                        default=False,
                        help='Do not tag hashtags.')
    parser.add_argument('--no-userref-tag', '-e',
                        dest='no_userref',
                        action='store_true',
                        default=False,
                        help='Do not tag user references.')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


def tag_url(text, tag=u'URL'):
    # source: http://stackoverflow.com/questions/6883049/regex-to-find-urls-in-string-in-python
    # test: import re; re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' , u'URL', 'ahttp:/www.uol.com.br ahttp://www.uol.com.br https://255.255.255.255/teste http://www.255.1.com/outroteste a a a ')
    #return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' , u'URL', text)

    # Thiago Cavalcante's approach
    return re.sub('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)', tag, text)


def tag_userref(text, tag=u'REF'):
    # rational: a username must start with a '@' and have unlimited occurences of letters, numbers and underscores.
    # test: import re; re.sub('(?<!\S)@[0-9a-zA-Z_]{1,}(?![0-9a-zA-Z_])', u'REF', '@user @us3r @1user @1234567890123456 @_0334 @vser @_ @1 @faeeeec-cas caece ce ce asdcc@notuser ewdede-@dqwec email@some.com.br @ @aaa')
    #return re.sub('(?<!\S)@[0-9a-zA-Z_]{1,}(?![0-9a-zA-Z_])', u'REF', text)

    # Thiago Cavalcante's approach
    return re.sub('@[^\s]+', tag, text)


def tag_hashtag(text, tag=u'TAG'):
    # rationale: https://support.twitter.com/articles/49309
    # test: import re; re.sub('(?<!\S)(#[0-9a-zA-Z_-]+)(?![0-9a-zA-Z_-])', u'TAG', '#anotherhash #123 #a123 a not#hash #[]aaa #avbjd')
    #return re.sub('(?<!\S)(#[0-9a-zA-Z_-]+)(?![0-9a-zA-Z_-])', u'TAG', text)

    # Thiago Cavalcante's approach
    return re.sub('#[a-zA-Z]+', tag, text)


def tag_date(text, tag=u'DAT'):
    # rationale: a date is a two or three blocks of digits separated by a slash.
    # test: import re; re.sub(('(?<!\S)('
    #                                       '[0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9]\s?[/-]\s?[0-9]{1,4}|'     # DD/MM/YYYY or MM/DD/YYYY
    #                                       '[0-1]?[0-9]\s?[/-]\s?[0-9]{1,4}|'                          # MM/YYYY
    #                                       '[0-9]{1,4}\s?[/-]\s?[0-1]?[0-9]|'                          # YYYY/MM
    #                                       '[0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9]'                          # DD/MM or MM/DD
    #                       '            )(?![0-9a-zA-Z])'
    #                           ), u'DAT', '23/12/1977 12 - 23- 2014 25-10 12 / 23 09/2013 1999 - 02 90/12 a12/94 12/31. 12/31a 12-31')
    #return re.sub(('(?<!\S)('
    #                            '[0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9]\s?[/-]\s?[0-9]{1,4}|'     # DD/MM/YYYY or MM/DD/YYYY
    #                            '[0-1]?[0-9]\s?[/-]\s?[0-9]{1,4}|'                          # MM/YYYY
    #                            '[0-9]{1,4}\s?[/-]\s?[0-1]?[0-9]|'                          # YYYY/MM
    #                            '[0-3]?[0-9]\s?[/-]\s?[0-3]?[0-9]'                          # DD/MM or MM/DD
    #                       ')(?![0-9a-zA-Z])'
    #               ), u'DAT', text)

    # Thiago Cavalcante's approach
    return re.sub('[0-9]?[0-9][-/][0-9]?[0-9]([-/][0-9][0-9][0-9][0-9])?', tag, text)


def tag_time(text, tag=u'TIM'):
    # rationale: a time is one or two digits followed by a colon and one or two more digits followed by an optional seconds block. An optional AM/PM suffix can also occur.
    # test: import re; re.sub('(?<!\S)([0-2]?[0-9]:[0-5]?[0-9](:[0-5]?[0-9])?\s?([A|P]M)?)(?![0-9a-zA-Z])', u'TIM', '00:00 AM 1:01PM 2:2 pm 01:02:03 Am 01:02. 03:12! 03:14a bbb 60:60 3:40am', flags=re.IGNORECASE)
    #return re.sub('(?<!\S)([0-2]?[0-9]:[0-5]?[0-9](:[0-5]?[0-9])?\s?([A|P]M)?)(?![0-9a-zA-Z])', u'TIM', text, flags=re.IGNORECASE)

    # Thiago Cavalcante's approach
    return re.sub('[0-9]?[0-9]:[0-9]?[0-9](:[0-9]?[0-9])?', tag, text)


def tag_number(text, tag=u'NUM'):
    # rationale: a number is a group of consecutive digits, comma and points, prefixed by a optional plus/minus. Obs: expected very few false positives
    # test: import re; re.sub('(?<!\S)([+-]?[0-9.,]*[0-9])(?![0-9a-zA-Z+-])', u'NUM', '98.786    123 123.1 345,2 32, 56. .92 ,34 100,000.00 +11,3 -10 10? 10! 1,1..2 1-1 1+1 dadcd12  89hjgj tt.bt.65bnnn 98,3')
    #return re.sub('(?<!\S)([+-]?[0-9.,]*[0-9])(?![0-9a-zA-Z+-])', u'NUM', text)

    # rationale: a number is a group of three possibilities: 1) a leading digits followed by point/comma and optional decimal digits; 2) leading comma/point followed by digits; 3) numbers without comma/point
    # test: import re; re.sub('(?<!\S)([0-9]+[,.][0-9]*|[,.][0-9]+|[0-9]+)(?=\s|$)', u'NUM', '98.786    123 123.1 345,2 32, 56. .92 ,34 dadcd12  89hjgj tt.bt.65bnnn 98,3')
    # return re.sub('(?<!\S)([0-9]+[,.][0-9]*|[,.][0-9]+|[0-9]+)(?=\s|$)', u'NUM', text)

    # Thiago Cavalcante's approach
    return re.sub('[0-9]+', tag, text)


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting tagging data ...',
                           '\n\tdataset directory = ', args.dataset_directory,
                           '\n\tdestination directory = ', args.destination_directory,
                           '\n\ttag numbers = ', str(not args.no_number),
                           '\n\ttag dates = ', str(not args.no_date),
                           '\n\ttag times = ', str(not args.no_time),
                           '\n\ttag URLs= ', str(not args.no_url),
                           '\n\ttag hashtags = ', str(not args.no_hashtag),
                           '\n\ttag user references = ', str(not args.no_userref),
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Creating output directory ...')
    if os.path.exists(args.destination_directory):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.destination_directory)

    logging.info('Tagging tweets ...')
    users_dirs = glob.glob(os.sep.join([args.dataset_directory, '[0-9]*']))
    i = 0                           # processing feedback
    num_users = len(users_dirs)     # processing feddback
    logging.info('Processing files ...')
    for user_dir in users_dirs:
        i += 1
        sys.stdout.write('\t%d/%d files processed\r' % (i,num_users))
        with open(os.sep.join([user_dir, 'tweets.json']), mode='rt', encoding='ascii') as fd:
            tweets = json.load(fd)
        tagged_texts = []
        for tweet in tweets:
            tagged = tweet['text']
            if not args.no_url:
                tagged = tag_url(tagged)
            if not args.no_userref:
                tagged = tag_userref(tagged)
            if not args.no_hashtag:
                tagged = tag_hashtag(tagged)
            if not args.no_date:
                tagged = tag_date(tagged)
            if not args.no_time:
                tagged = tag_time(tagged)
            if not args.no_number:
                tagged = tag_number(tagged)
            logging.debug('Original message: ' + tweet['text'])
            logging.debug('Tagged message: ' + tagged)
            tagged_texts.append({'id':tweet['id'], 'text':tagged})

        os.makedirs(os.sep.join([args.destination_directory, os.path.basename(user_dir)]))
        with open(os.sep.join([args.destination_directory, os.path.basename(user_dir), 'tweets.json']), mode='wt', encoding='ascii') as fd:
            json.dump(tagged_texts, fd, sort_keys=True, ensure_ascii=True)
    logging.info('Finished.')
