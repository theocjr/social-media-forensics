#!/usr/bin/env python3


import argparse
import logging
import pprint
import sklearn
import sklearn.datasets
import sklearn.externals
import numpy
import xgboost


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument('--training-file', '-t',
                        dest='training_file',
                        required=True,
                        help='File in LBSVM format with the training data.')
    parser.add_argument('--validation-file', '-v',
                        dest='validation_file',
                        required=True,
                        help='File in LBSVM format with the validation data.')
    parser.add_argument('--test-file', '-e',
                        dest='test_file',
                        required=True,
                        help='File in LBSVM format with the test data.')
    parser.add_argument('--destination-file', '-r',
                        dest='destination_file',
                        required=True,
                        help='File where the grid search results will be stored.')
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
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')


    logging.info('Configuration parameters:\n' + pprint.pformat(vars(args)))

    logging.info('Reading data ...')
    x_train, y_train = sklearn.datasets.load_svmlight_file(args.training_file)
    x_train = x_train.todense()
    x_valid, y_valid = sklearn.datasets.load_svmlight_file(args.validation_file)
    x_valid = x_valid.todense()
    x_test, y_test = sklearn.datasets.load_svmlight_file(args.test_file)
    x_test = x_test.todense()
    x_train_valid = numpy.vstack((x_train, x_valid))
    y_train_valid = numpy.concatenate((y_train, y_valid))


    logging.info('Starting grid serach ...')
    params = {'objective':'multi:softmax', 'num_class':50, 'verbosity':1}
    hyper_params = {
        'learning_rate':[0.3], # 0.01-0.3
        'max_depth':[3, 7, 10], # 3-10
        'min_child_weight': [1, 3, 6], # 1-6
        'gamma':[0, 0.25, 0.5], # 0-0.5
        'subsample':[0.5, 0.75, 1], # 0.5-1
        'colsample_bytree':[0.5, 0.75, 1], # 0.5-1
        'n_estimators':[50], 
        'n_jobs': [-2],
    }
#    hyper_params = {
#        'learning_rate':[0.3], # 0.01-0.3
#        'max_depth':[3, 7, 10], # 3-10
#        'min_child_weight': [1, 3, 6], # 1-6
#        'gamma':[0, 0.25, 0.5], # 0-0.5
#        'subsample':[0.5, 0.75, 1], # 0.5-1
#        'colsample_bytree':[0.5, 0.75, 1], # 0.5-1
#        'n_estimators':[50], 
#        'n_jobs': [-2],
#    }

    model = xgboost.XGBClassifier(**params)
    param_searcher = sklearn.model_selection.GridSearchCV(model, hyper_params, cv=2, n_jobs=50, verbose=2)
    param_searcher.fit(x_train_valid, y_train_valid)

    logging.info('Best parameter set found:\n{}'.format(pprint.pformat(param_searcher.best_params_)))
    logging.info('Best validation score: {}%'.format(param_searcher.best_score_ * 100.0))
    test_score = param_searcher.best_estimator_.score(x_test, y_test)
    logging.info('Test score with the best estimator: {}%'.format(test_score * 100.0))

    logging.info('Saving grid search data ...')
    sklearn.externals.joblib.dump(param_searcher, args.destination_file)

    logging.info('Finished.')
