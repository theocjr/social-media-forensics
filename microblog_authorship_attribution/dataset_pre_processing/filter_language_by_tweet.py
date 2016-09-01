#!/usr/bin/env python


"""
Code for reading authors' tweets filenames and decide which is the
    language based on an API of language detection (guess_language in this
    case). If the language is different from the one specified in the 
    --language command-line option, the tweet is not copied to the output file.
The output files are renamed to have the number of tweets selected prefixed in
    their filenames.
"""


import argparse
import logging
import sys
import os
import glob
import messages_persistence
import traceback


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
    parser.add_argument('--language-detection-module-dir', '-m',
                        dest='lang_mod_dir',
                        required=True,
                        help='Directory where the language detection module resides.')
    parser.add_argument('--language', '-l',
                        dest='language',
                        default='English',
                        help='Language used to filter the tweets. Defaults to English.')
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

    logging.info(''.join(['Starting filtering language ...',
                           '\n\tsource directory data = ', args.source_dir_data,
                           '\n\tdestination directory = ', args.dest_dir,
                           '\n\tlanguage detection module directory = ', args.lang_mod_dir,
                           '\n\tlanguage = ', args.language,
                           '\n\tdebug = ', str(args.debug),
                         ]))

    sys.path.append(args.lang_mod_dir)
    import guess_language

    logging.info('Creating output directories ...')
    if os.path.exists(args.dest_dir):
        logging.error(''.join(['Destination directory ', args.dest_dir, ' already exists. Quitting ...']))
        sys.exit(1)
    os.makedirs(args.dest_dir)

    logging.info('Filtering tweets by language ...')
    author_filenames = glob.glob(''.join([args.source_dir_data, os.sep, '*.dat']))
    num_files = len(author_filenames)   # processing feedback
    i = 0                               # processing feedback
    for author_filename in author_filenames:
        sys.stdout.write(''.join(['\t', str(i), '/', str(num_files), ' files processed\r']))
        i += 1
        logging.debug(''.join(['Processing ', author_filename, ' file ...']))
        messages = messages_persistence.read(author_filename)
        messages_filtered = []
        for message in messages:
            logging.debug(''.join(['Detecting language for tweet: ', message['tweet']]))
            try:        # code guess-language breaks for some tweets
                detected_language = guess_language.guessLanguageName(message['tweet'])
            except Exception as e:
                logging.warning('guess-language library error in detecting language for tweet: ' + message['tweet'])
                logging.warning('Exception message: ' + str(e))
                logging.warning('Exception stack trace:')
                traceback.print_tb(sys.exc_info()[2])
                detected_language = None
            if detected_language:
                logging.debug(''.join(['\tLanguage \'', detected_language, '\' detected.']))
                if detected_language == args.language:
                    messages_filtered.append(message)
            else:
                logging.warning('No language detected for tweet: ' + message['tweet'])

        destination_filename = ''.join([args.dest_dir, os.sep, str(len(messages_filtered)).zfill(5), '_', os.path.basename(author_filename)])      # the destination name is the original filename prefixed with the number of tweets
        logging.debug(''.join(['Saving ', destination_filename, ' file ...']))
        messages_persistence.write(messages_filtered, 'full', destination_filename)
        
    logging.info('Finishing ...')
