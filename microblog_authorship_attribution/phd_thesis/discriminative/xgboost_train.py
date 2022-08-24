#!/usr/bin/env python3


import argparse
import logging
import pprint
import os
import sys
import sklearn
import sklearn.externals
import numpy
import xgboost


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument('--training_file', '-t',
                        required=True,
                        help='File in LBSVM format with the training data.')
    parser.add_argument('--validation_file', '-v',
                        required=True,
                        help='File in LBSVM format with the validation data.')
    parser.add_argument('--test_file', '-e',
                        required=True,
                        help='File in LBSVM format with the test data.')
    parser.add_argument('--destination_directory', '-o',
                        required=True,
                        help='Directory where the output files will be written.')
    parser.add_argument('--out_of_core',
                        action='store_true',
                        default=False,
                        help='Use of external memory to handle data. Default = False.')
    parser.add_argument('--resume_training_model', '-m',
                        help='Model in sklearn.externals.joblib file format to resume training.')
    parser.add_argument('--resume_training_configuration', '-c',
                        help='Configuration of the model in sklearn.externals.joblib file format to resume training. If specified, some configuration values in this file overwrites the values passed as command-line options (see function update_xgboost_configuration for more details).')
    parser.add_argument('--num_trees',
                        type=int,
                        default=10,
                        help='XGBoost parameter controlling the number of trees built. Default = 10.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.3,
                        help='XGBoost learning rate parameter. Default = 0.3 .')
    parser.add_argument('--max_depth',
                        type=int,
                        default=6,
                        help='XGBoost parameter controlling maximum depth of a tree. Default = 6.')
    parser.add_argument('--min_child_weight',
                        type=float,
                        default=1.0,
                        help='XGBoost parameter minimum sum of instance weight needed in a child. Default = 1.0 .')
    parser.add_argument('--min_split_loss',
                        type=float,
                        default=0.0,
                        help='XGBoost parameter controlling the minimum loss reduction required to make a further partition on a leaf node of the tree. Default = 0.0.')
    parser.add_argument('--subsample',
                        type=float,
                        default=1.0,
                        help='XGBoost parameter controlling the ratio of the training instances. Default = 1.0 .')
    parser.add_argument('--colsample_bytree',
                        type=float,
                        default=1.0,
                        help='XGBoost parameter controlling the subsample ratio of columns when constructing each tree. Default = 1.0 .')
    parser.add_argument('--reg_lambda',
                        type=float,
                        default=1.0,
                        help='XGBoost L2 regularization on the weights. Default = 1.0 .')
    parser.add_argument('--reg_alpha',
                        type=float,
                        default=0.0,
                        help='XGBoost L1 regularization on the weights. Default = 0.0 .')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information. Default = False.')

    args = parser.parse_args()
    if bool(args.resume_training_model) != bool(args.resume_training_configuration):
        parser.error('If --resume_training_model is specified, --resume_training_configuration also must be (and vice-versa).')
    return args


def update_xgboost_configuration(resume_args_filename, args):
    resume_args = sklearn.externals.joblib.load(resume_args_filename)
    args.learning_rate      = resume_args.learning_rate
    args.max_depth          = resume_args.max_depth
    args.min_child_weight   = resume_args.min_child_weight
    args.min_split_loss     = resume_args.min_split_loss
    args.subsample          = resume_args.subsample
    args.colsample_bytree   = resume_args.colsample_bytree
    args.reg_lambda         = resume_args.reg_lambda
    args.reg_alpha          = resume_args.reg_alpha
    return


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    resume_training_model = None
    if args.resume_training_configuration:
        logging.info('Loading previous XGBoost model and configuration ...')
        update_xgboost_configuration(args.resume_training_configuration, args)
        resume_training_model = sklearn.externals.joblib.load(args.resume_training_model)
    logging.info('Configuration parameters:\n' + pprint.pformat(vars(args)))

    logging.info('Creating output directory ...')
    if os.path.exists(args.destination_directory):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.destination_directory)
    sklearn.externals.joblib.dump(args, os.sep.join([args.destination_directory, 'configuration.skl']))

    logging.info('Reading data ...')
    train_cache = ''
    valid_cache = ''
    test_cache = ''
    if args.out_of_core:
        train_cache = ''.join(['#', args.destination_directory, os.sep, 'training.cache'])
        valid_cache = ''.join(['#', args.destination_directory, os.sep, 'validation.cache'])
        test_cache = ''.join(['#', args.destination_directory, os.sep, 'test.cache'])
    train = xgboost.DMatrix(args.training_file + train_cache)
    valid = xgboost.DMatrix(args.validation_file + valid_cache)
    test = xgboost.DMatrix(args.test_file + test_cache)

    if resume_training_model:
        logging.info('Starting training from an existing model ...')
    else:
        logging.info('Starting training ...')
    hyper_params = {
        'objective':        'multi:softmax',
        'num_class':        50,
        'verbosity':        2 if args.debug else 1,
        'eta':              args.learning_rate,
        'max_depth':        args.max_depth,
        'min_child_weight': args.min_child_weight,
        'gamma':            args.min_split_loss,
        'subsample':        args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'lambda':           args.reg_lambda,
        'alpha':            args.reg_alpha,
    }
    evals_result = {}
    model = xgboost.train(hyper_params,
                          train,
                          num_boost_round=args.num_trees,
                          evals=[(train, 'train'), (valid, 'eval')],
                          early_stopping_rounds=10,
                          evals_result=evals_result,
                          verbose_eval=True,
                          xgb_model=resume_training_model,
                         )

    logging.info('Saving model and results ...')
    sklearn.externals.joblib.dump(model, os.sep.join([args.destination_directory, 'xgboost_model.skl']))
    sklearn.externals.joblib.dump(evals_result, os.sep.join([args.destination_directory, 'xgboost_evals_result.skl']))

    logging.info('Error score over test data: {}'.format(model.eval(test)))
    
    if args.out_of_core:
        logging.info('Deleting cache files ...')
        train_cache = train_cache[1:]
        os.remove(train_cache)
        os.remove(train_cache + '.row.page')
        os.remove(train_cache + '.sorted.col.page')
        valid_cache = valid_cache[1:]
        os.remove(valid_cache)
        os.remove(valid_cache + '.row.page')
        test_cache = test_cache[1:]
        os.remove(test_cache)
        os.remove(test_cache + '.row.page')
    
    logging.info('Finished.')
