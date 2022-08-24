#!/usr/bin/env python3
"""
"""

import argparse
import logging
import pprint
import os
import sys
import sklearn
import csv
import shutil


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument('--source_dir_data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the input data files (in CSV format) are stored.')
    parser.add_argument('--destination_directory', '-b',
                        dest='destination_directory',
                        required=True,
                        help='Directory where the mapped dataset will be stored.')
    parser.add_argument('--ngrams_source', '-s',
                        dest='ngrams_source',
                        help='Filename containing the source n-grams mapping.')
    parser.add_argument('--ngrams_dest', '-e',
                        dest='ngrams_dest',
                        help='Filename containing the destination n-grams mapping.')
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

    logging.info('Configuration parameters:\n' + pprint.pformat(vars(args)))

    logging.info('Creating output directory ...')
    if os.path.exists(args.destination_directory):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.destination_directory)

    logging.info('Loading n-grams dictionaries ...')
    logging.info('\tSource n-grams ...')
    ngrams_source = sklearn.externals.joblib.load(args.ngrams_source)
    ngrams_source_inv = { str(index): gram for gram, index in ngrams_source.items() }
    logging.info('\tDestination n-grams ...')
    ngrams_dest = sklearn.externals.joblib.load(args.ngrams_dest)

    logging.info('Transforming files ...')
    for filename in ('training.csv', 'valid.csv', 'test.csv'):
        with open(os.sep.join([args.source_dir_data, filename]), mode='rt', encoding='ascii') as r_fd, open(os.sep.join([args.destination_directory, filename]), mode='xt', encoding='ascii', newline='') as w_fd:
            logging.info('\tFile {} ...'.format(filename))
            csv_writer = csv.writer(w_fd)
            csv_writer.writerow(r_fd.readline().rstrip().split(','))    # header
            num_ngrams = 0                                              # stats
            unknown_ngrams = set()
            unknown_occurs = 0
            for row in r_fd:
                fields = row.rstrip().split(',')
                w_row = [ fields[0] ]                                   # tweet ID
                num_ngrams += len(fields) - 1
                for idx in range(1, len(fields)):
                    ngram = ngrams_source_inv[fields[idx]]
                    if ngram in ngrams_dest:
                        w_row.append(ngrams_dest[ngram])
                    else:
                        w_row.append(ngrams_dest['\0'])
                        unknown_occurs += 1
                        unknown_ngrams.add(ngram)
                        logging.debug('\t\t{} n-gram not found.'.format(ngram))
                csv_writer.writerow(w_row)
        logging.info('\t{} file translated. {}/{} n-grams occurences not found. {} unique unknown n-grams.'.format(filename, unknown_occurs, num_ngrams, len(unknown_ngrams)))

        logging.info('Copying label files ...')
        for filename in ('training_lbl.csv', 'valid_lbl.csv', 'test_lbl.csv'):
            shutil.copy2(os.sep.join([args.source_dir_data, filename]), args.destination_directory)

    logging.info('Finished.')
