#!/usr/bin/env python


"""
Program to trigger a Random Forest classifier based on FEST code. Its input are the feature vectors output by feature_vectors_generator.py code. The FEST code doesn't perform a multi-class classification (only binary) so a model for each author is learned.

Logic in pseudo-code
    1 - For each run (directory named run_XXX in input directory)
        1.1 - For each fold (directory named fold_XX in run directory)
            1.1.1 - Read training and test feature vectors.
            1.1.2 - For each author
                1.1.1.1 - Build positive and negative sets from training data (ratio of 3)
                1.1.1.2 - Train a model using positive and negative sets
                1.1.1.3 - Test the model using test data
                1.1.1.4 - Register author prediction
            1.1.3 - Account fold accuracy
         1.2 - Account run accuracy
     2 - Account final accuracy


Obs. Premise: the samples of each author in the training set are grouped together.
"""


import argparse
import logging
import glob
import os
import sys
import sklearn.datasets
import numpy
import random


def command_line_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir-data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the feature vectors files are stored.')
    parser.add_argument('--output-dir', '-b',
                        dest='output_dir',
                        required=True,
                        help='Directory where the output files will be written.')
    parser.add_argument('--number-trees', '-e',
                        dest='num_trees',
                        type=int,
                        default=500,
                        help='Number of trees in the Random Forest classifier. Default = 500.')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')
    return parser.parse_args()


if  __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info(''.join(['Starting the RF classification ...',
                           '\n\tsource directory data = ', args.source_dir_data,
                           '\n\toutput directory = ', args.output_dir,
                           '\n\tnumber of trees = ', str(args.num_trees),
                           '\n\tdebug = ', str(args.debug),
                         ]))


    logging.info('Creating output directory ...')
    if os.path.exists(args.output_dir):
        logging.error('Output directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.output_dir)

    logging.info('Learning and running classifiers ...')
    script_dir = os.path.dirname(os.path.realpath(__file__))

    run_accuracies = []
    for run in sorted(glob.glob(os.sep.join([args.source_dir_data, 'run_*']))):
        logging.debug('Run ' + run)
        with open(os.sep.join([run, 'sampled_authors.txt'])) as fd:
            num_sampled_authors = len(fd.readlines())

        fold_accuracies = []
        for fold in sorted(glob.glob(os.sep.join([run, 'fold_*']))):
            logging.debug('\tFold ' + fold)

            logging.debug('\tReading training data ...')
            with open(os.sep.join([fold, 'pmsvm_train.dat'])) as fd:
                training_samples = fd.readlines()

            logging.debug('\tReading test data ...')
            x_test, y_test = sklearn.datasets.load_svmlight_file(os.sep.join([fold, 'pmsvm_test.dat']), zero_based=False)
            predictions = numpy.zeros( (y_test.shape[0], num_sampled_authors) )

            train_samples_per_author = len(training_samples) / num_sampled_authors
            for author_id in range(num_sampled_authors):
                logging.debug(''.join(['\t\tAuthor ', str(author_id), ' ...']))
                author_dir = os.sep.join([args.output_dir, os.path.basename(run), os.path.basename(fold), 'author_' + str(author_id)])
                os.makedirs(author_dir)
                author_start_idx = author_id * train_samples_per_author
                author_end_idx = author_start_idx + train_samples_per_author
                
                logging.debug('\t\tBuilding positive class ...')
                positive_samples_idxs = range(author_start_idx, author_end_idx)
                positive_samples = []
                for idx in positive_samples_idxs:
                    fields = training_samples[idx].split()
                    fields[0] = '1'         # positive class = 1
                    positive_samples.append(' '.join(fields))

                logging.debug('\t\tBuilding negative class ...')
                negative_samples_idxs = range( 0, author_start_idx ) + range( author_end_idx, len(training_samples) )
                random.shuffle(negative_samples_idxs)
                negative_samples_idxs = negative_samples_idxs[ : train_samples_per_author * 3]      # negative class size should be three times bigger than the positive
                negative_samples = []
                for idx in negative_samples_idxs:
                    fields = training_samples[idx].split()
                    fields[0] = '0'         # negative class = 0
                    negative_samples.append(' '.join(fields))

                logging.debug('\t\tWriting training data ...')
                with open(os.sep.join([author_dir, 'train.dat']), mode='w+') as fd:
                    fd.write('\n'.join(positive_samples))
                    fd.write('\n'.join(negative_samples))
                    fd.write('\n')

                logging.debug('\t\tLearning RF ...')
                ret_code = os.system(''.join([script_dir, os.sep, 'fest', os.sep, 'festlearn',  # executable
                                        ' -c 3 -t ', str(args.num_trees),                       # options
                                        ' ', author_dir, os.sep, 'train.dat',                   # training data file
                                        ' ', author_dir, os.sep, 'rd_model.dat',                # model data file
                                             ])
                                    )
                if ret_code != 0:
                    logging.error(''.join(['Error learning RF classifier for author ', str(author_id), ', fold ',  fold, '. Error code = ', str(ret_code), ' . Exiting ...']))
                    sys.exit(1)

                logging.debug('\t\tClassifying ...')
                ret_code = os.system(''.join([script_dir, os.sep, 'fest', os.sep, 'festclassify',   # executable
                                        ' ', fold, os.sep, 'pmsvm_test.dat',                        # test data file
                                        ' ', author_dir, os.sep, 'rd_model.dat',                    # model data file
                                        ' ', author_dir, os.sep, 'rd_prediction.dat',               # prediction data file
                                             ])
                                    )
                if ret_code != 0:
                    logging.error(''.join(['Error classifying test data against RF for author ', str(author_id), ', fold ',  fold, '. Error code = ', str(ret_code), ' . Exiting ...']))
                    sys.exit(1)
                with open(os.sep.join([author_dir, 'rd_prediction.dat'])) as fd:
                    author_rf_predictions = fd.readlines()
                for idx in range(len(author_rf_predictions)):
                    predictions[idx, author_id] = float(author_rf_predictions[idx])

            fold_predictions = numpy.argmax(predictions, axis=1)
            fold_accuracies.append( 100.0 * ( 1.0 - ( numpy.count_nonzero(fold_predictions - y_test) / float(len(y_test)) ) ) )
            logging.info(''.join(['\tFold accuracy: ', str(fold_accuracies[-1]), '%']))

        run_accuracies.append( numpy.mean(fold_accuracies) )
        logging.info(''.join(['Run accuracy: ', str(run_accuracies[-1]), '%']))

    logging.info(''.join(['Final accuracy: ', str(numpy.mean(run_accuracies)), '%']))

    logging.info('Finishing ...')
