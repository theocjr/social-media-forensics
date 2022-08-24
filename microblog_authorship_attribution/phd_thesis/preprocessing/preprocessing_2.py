#!/usr/bin/env python3
#TODO: identify positive/negative emoticons and tag them


import argparse
import logging
import pprint
import os
import sys
import glob
import re
import string
import html
from functools import reduce, partial
import json

import emoji
from wordsegment import load, segment


URL_REGEX = '((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)|(http\S*\u2026)'  # \u2026 is the ellipsis Unicode char Twitter puts in the end of long URLs
URL_TAG = u'<http>'

USER_REGEX = '@[^\s]+'
USER_TAG = u'<user>'

NUMBER_REGEX = '[0-9]([0-9,.]*[0-9])?'
NUMBER_TAG = u'<number>'

HASHTAG_REGEX = '#[^#\s]+'      # bad-formatted hashtags with hash char (#) inside causes an infinite loop bug
HASHTAG_START_TAG = u'<hashtag>'
HASHTAG_END_TAG = u'</hashtag>'


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--input_directory', '-i',
                        required=True,
                        help='Directory of authors tweets.')
    parser.add_argument('--output_directory', '-o',
                        required=True,
                        help='Output directory.')
    parser.add_argument('--tag_url',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--tag_user',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--tag_number',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--tag_hashtag',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--demojize',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--textify_emoji',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--mention_limit',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--punc_limit',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--lower_hashtag',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--segment_hashtag',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--lower_case',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--add_capital_signs',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information.')

    args = parser.parse_args()
    if args.textify_emoji and not args.demojize:
        print('ERROR - --textify_emoji is meaningless without --demojize.', file=sys.stderr)
        sys.exit(1)
    if args.lower_case and args.add_capital_signs:
        print('ERROR - --lower_case and --add_capital_signs are incompatible. You must choose at most one option.', file=sys.stderr)
        sys.exit(1)
    if args.tag_hashtag and (args.lower_hashtag or args.segment_hashtag):
        print('ERROR - --tag_hashtag is incompatible with --lower_hashtag or --segment_hashtag.', file=sys.stderr)
        sys.exit(1)
    return args


def compose(*funcs):
    """" Compose functions so that they are applied in chain. """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs[::-1])


def normalize_quotes(text):
    text = re.sub(r'[\u2018\u2019]', '\'', text)    # single quotes
    text = re.sub(r'[\u201C\u201D]', '\"', text)    # double quotes
    return text


def tag(text, regex, tag):
    tokens = []
    for token in text.split():
        if re.fullmatch(regex, token):
            tokens.append(tag)
        else:
            tokens.append(token)
    return ' '.join(tokens)


def replace_emojis(sent):
    """ e.g. smiling emoticon -> ^smiley_face$ """
    return emoji.demojize(sent, delimiters=('^','$'))


def textify_emojis(sent):
    """ e.g. ^smiley_face$ -> smiley face"""
    #return re.sub(':[\S]+:', lambda match: match.group().replace('_', ' ').replace('-', ' ').replace(':', ''), sent)
    #ret = re.sub(':[\S]+:', lambda match: match.group().replace('_', ' ').replace(':', ''), sent)
    #return '<emoji> ' + ret + ' </emoji>'
    return re.sub('\^[\S]+\$', lambda match: match.group().replace('_', ' ').replace('-', ' ').replace('^', ' <emoji> ').replace('$', ' </emoji> '), sent)


def _limit_pattern(sent, pattern, keep_num):
    if pattern in string.punctuation:
        re_pattern = re.escape(pattern)
    else:
        re_pattern = f'(({pattern})[\s]*)'
        pattern = pattern + ' '
    pattern_regex = re_pattern + '{' + str(keep_num+1) + ',}'
    return re.sub(pattern_regex, lambda match: pattern * keep_num, sent)


def limit_mentions(sent, keep_num):
    return _limit_pattern(sent, '@user', keep_num)


def limit_punctuations(sent, keep_num):
    puncs = ['!', '?', '.']
    for p in puncs:
        sent = _limit_pattern(sent, p, keep_num)
    return sent


def lower_hashtags(sent):
    """ e.g.  #MAGA -> #maga """
    return re.sub(HASHTAG_REGEX, lambda match: match.group().lower(), sent)


def segment_hashtags(sent):
    """ e.g. #MakeAmericaGreatAgain -> make america great again"""
    #return re.sub('#[\S]+', lambda match: ' '.join(segment(match.group())), sent)
    return re.sub(HASHTAG_REGEX, lambda match: '{} {} {}'.format(HASHTAG_START_TAG, ' '.join(segment(match.group())), HASHTAG_END_TAG), sent)
    #return 'HASHTAG_START_TAG ' + ret + ' HASHTAG_END_TAG'


def add_capital_signs(text):
    def _has_cap(token):
        return token.lower() != token and token.upper() != token
    def _all_cap(token):
        return token.lower() != token and token.upper() == token
    tokens = text.split()
    tokens = ['<has_cap> ' + t if _has_cap(t) else t for t in tokens]
    tokens = ['<all_cap> ' + t if _all_cap(t) else t for t in tokens]
    return ' '.join(tokens).lower()


def build_preprocess(tag_url,
                     tag_user,
                     tag_number,
                     tag_hashtag,
                     demojize,
                     textify_emoji,
                     mention_limit,
                     punc_limit,
                     lower_hashtag,
                     segment_hashtag,
                     lower_case,
                     add_capital_signs,
                    ):
    if textify_emoji and not demojize:
        raise Exception('textify_emoji is meaningless without demojize.')
    if lower_case and add_capital_signs:
        raise Exception('lower_case and add_capital_signs are incompatible. You must choose at most one option.')
    if tag_hashtag and (lower_hashtag or segment_hashtag):
        raise Exception('tag_hashtag is incompatible with lower_hashtag or segment_hashtag.')


    funcs = [
        html.unescape,
        normalize_quotes,
    ]
    if tag_url:
        funcs.append(partial(tag, regex=URL_REGEX, tag=URL_TAG))
    if tag_user:
        funcs.append(partial(tag, regex=USER_REGEX, tag=USER_TAG))
    if tag_number:
        funcs.append(partial(tag, regex=NUMBER_REGEX, tag=NUMBER_TAG))
    if tag_hashtag:
        funcs.append(partial(tag, regex=HASHTAG_REGEX, tag=HASHTAG_START_TAG))
    if demojize:
        funcs.append(replace_emojis)
    if textify_emoji:
        funcs.append(textify_emojis)
    if mention_limit > 0:
        funcs.append(partial(limit_mentions, keep_num=mention_limit))
    if punc_limit > 0:
        funcs.append(partial(limit_punctuations, keep_num=punc_limit))
    if lower_hashtag:
        funcs.append(lower_hashtags)
    if segment_hashtag:
        load()
        funcs.append(segment_hashtags)
    if lower_case:
        funcs.append(str.lower)
    if add_capital_signs:
        funcs.append(add_capital_signs)
    return compose(*funcs)


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    logging.info('Starting preprocessing text with the following parameters:\n{}'.format(pprint.pformat(vars(args))))

    logging.info('Creating destination directory {} ...'.format(args.output_directory))
    if os.path.exists(args.output_directory):
        logging.error('Output directory already exists. Quitting ...')
        sys.exit(1)
    os.mkdir(args.output_directory)

    preprocessor = build_preprocess(tag_url=args.tag_url,
                                    tag_user=args.tag_user,
                                    tag_number=args.tag_number,
                                    tag_hashtag=args.tag_hashtag,
                                    demojize=args.demojize,
                                    textify_emoji=args.textify_emoji,
                                    mention_limit=args.mention_limit,
                                    punc_limit=args.punc_limit,
                                    lower_hashtag=args.lower_hashtag,
                                    segment_hashtag=args.segment_hashtag,
                                    lower_case=args.lower_case,
                                    add_capital_signs=args.add_capital_signs,
                                   )

    author_dirnames = sorted(glob.glob(os.path.join(args.input_directory, '*')))
    for author_dirname in author_dirnames:
        basename = os.path.basename(author_dirname)
        logging.debug('Processing author {} ...'.format(basename))
        with open(os.path.join(author_dirname, 'tweets.json'), mode='rt', encoding='utf-8') as fd:
            tweets = json.load(fd)
        output = []
        for tweet in tweets:
            processed = preprocessor(tweet['text'])
            if processed:
                output.append({ 'id':   tweet['id'],
                               'text':  processed,
                              })
        os.mkdir(os.path.join(args.output_directory, basename))
        with open(os.path.join(args.output_directory, basename, 'tweets.json'), mode='xt', encoding='utf-8') as fd:
            json.dump(output, fd)

    logging.info('Finished.')
