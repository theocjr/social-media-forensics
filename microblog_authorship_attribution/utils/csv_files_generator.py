#!/usr/bin/env python3


"""
Code to read the training, validation and test JSON files for each user and
    group them in a single file (one for training, one for validation and one
    for test data).
The output files are in CSV format.
"""


import argparse
import logging
import os
import sys
import glob
import json
import csv


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--source-dir-data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the tweets\' files are stored.')
    parser.add_argument('--output-directory', '-b',
                        dest='output_directory',
                        required=True,
                        help='Directory where the output files will be written.')
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

    logging.info(''.join(['Starting transforming data ...',
                           '\n\tsource directory data = ', args.source_dir_data,
                           '\n\toutput directory = ', args.output_directory,
                           '\n\tdebug = ', str(args.debug),
                         ]))

    logging.info('Creating output directory ...')
    if os.path.exists(args.output_directory):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.output_directory)

    train = []
    valid = []
    test = []
    author_dirs = glob.glob(os.sep.join([args.source_dir_data, '[0-9]*']))
    label_id = 0
    logging.info('Reading data ...')
    for author_dir in author_dirs:
        author_id = os.path.basename(author_dir)
        for filename, container in [('training.json', train), ('valid.json', valid), ('test.json', test)]:
            with open(os.sep.join([author_dir, filename]), mode='rt', encoding='ascii') as fd:
                messages = json.load(fd)
            for message in messages:
                container.append({'x': message['text'],
                                  'y': label_id,
                                  'tweet_id': message['id'],    # field used for tracking/debugging
                                  'author_id': author_id,       # field used for tracking/debugging
                                 })
        label_id += 1

    logging.info('Writing data ...')
    data_header = ['Twitter tweet id', 'data[0]', 'data[1]' , '...']
    label_header = ['Twitter user id', 'label'] 
    for x_filename, y_filename, x_header, y_header, container in [('training.csv', 'training_lbl.csv', data_header, label_header, train),
                                                                  ('valid.csv', 'valid_lbl.csv', data_header, label_header, valid),
                                                                  ('test.csv', 'test_lbl.csv', data_header, label_header, test),
                                                                 ]:
        with open(os.sep.join([args.output_directory, x_filename]), mode='xt', newline='') as x_fd, open(os.sep.join([args.output_directory, y_filename]), mode='xt', newline='') as y_fd:
            x_csv_writer = csv.writer(x_fd)
            y_csv_writer = csv.writer(y_fd)
            x_csv_writer.writerow(x_header)
            y_csv_writer.writerow(y_header)
            for row in container:
                x_csv_writer.writerow([row['tweet_id']] + row['x'])
                y_csv_writer.writerow([ row['author_id'] , row['y'] ])
    logging.info('CSV training, validation and test data files written.')
    logging.info('\tTraining data: {} samples.'.format(len(train)))
    logging.info('\tValidation data: {} samples.'.format(len(valid)))
    logging.info('\tTest data: {} samples.'.format(len(test)))
    logging.info('Finished.')
