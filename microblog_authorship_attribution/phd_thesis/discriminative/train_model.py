#!/usr/bin/env python
# Python 2 due to Theano/Lasagne

"""
"""


import argparse
import logging
import os
import sys
import pprint
import cPickle
import Dataset
import theano
import theano.tensor as T
import lasagne
import numpy
import time
import random

sys.path.append(os.sep.join(['.', 'DynamicCNN']))
import networks
import DCNN
import utils


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    # input/output
    parser.add_argument('--source_dir_data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the input data files (in CSV format) are stored. Minibatches generated will be stored in a sub-directory inside this directory.')
    parser.add_argument('--destination_directory', '-b',
                        dest='destination_directory',
                        required=True,
                        help='Directory where the outputs will be stored.')
    parser.add_argument('--parameters_filename', '-p',
                        dest='parameters_filename',
                        help='Filename containing the model parameters in Pickle format.')
    parser.add_argument('--updates_filename', '-u',
                        dest='updates_filename',
                        help='Filename containing the training updates in Pickle format.')
    parser.add_argument('--memory', '-m',
                        dest='memory',
                        action='store_true',
                        default=False,
                        help='If specified, minibatches are fully stored in memory. Default = False.')
    # training settings
    parser.add_argument('--batch_size',
                        type=int,
                        default=40,
                        help='Batch size. Default = 40.')                                       #changed from original
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        help='Learning rate. Default = 0.1 .')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs. Default = 10.')                                 # changed from original
    parser.add_argument('--max_minutes',
                        type=int,
                        default=0,
                        help='Maximum number of minutes to run the experiment. Default = 0 (no limit).')
    parser.add_argument('--valid_freq',
                        type=float,
                        default=0.0,
                        help='Validation frequency related to number of batches. If < 1.0, indicates the percentage of number of total batches. If >= 1.0, indicates the absolute number of batches after which the validation is performed. Deafult = 0.0 (validation after each epoch).')
    parser.add_argument('--adagrad_reset',
                        type=int,
                        default=5,
                        help='Resets the adagrad cumulative gradient after x epochs. If the value is 0, no reset will be executed. Default = 5.')
    # network paras
    parser.add_argument('--vocab_size',
                        required=True,
                        type=int,
                        help='Vocabulary size.')
    parser.add_argument('--output_classes',
                        required=True,
                        type=int,
                        help='Number of output classes.')
    parser.add_argument('--word_vector_size',
                        type=int,
                        default=48,
                        help='Word vector size. Default = 48.')
    parser.add_argument('--filter_size_conv_layers',
                        nargs='+',
                        type=int,
                        default=[10, 7],
                        help='List of sizes of filters at layer 1 and 2. Default=[10, 7].')
    parser.add_argument('--nr_of_filters_conv_layers',
                        nargs='+',
                        type=int,
                        default=[6, 12],
                        help='List of number of filters at layer 1 and 2. Default=[6, 12].')
    parser.add_argument('--activations',
                        nargs='+',
                        type=str,
                        default=['tanh', 'tanh'],
                        help='List of activation functions behind first and second conv layers. Default = [tanh, tanh]. Possible values are \'linear\', \'tanh\', \'rectify\' and \'sigmoid\'. ')
    parser.add_argument('--L2',
                        nargs='+',
                        type=float,
                        default=[0.0001/2,0.00003/2,0.000003/2,0.0001/2],
                        help='Fine-grained L2 regularization. 4 values are needed for 4 layers, namely for the embeddings layer, 2 conv layers and a final/output dense layer. Default = [0.0001/2,0.00003/2,0.000003/2,0.0001/2] .')
    parser.add_argument('--ktop',
                        type=int,
                        default=4,
                        help='K value of top pooling layer DCNN. Default = 4.')
    parser.add_argument('--dropout_value',
                        type=float,
                        default=0.5,
                        help='Dropout value after penultimate layer. Default = 0.5 .')
    # misc
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print debug information.')

    args = parser.parse_args()
    if args.updates_filename and not args.parameters_filename:
        parser.error('If --updates_filename is specified, --parameters_filename also must be.')
    return args


def load_model(network, parameters_filename, updates, updates_filename):
    logging.info('\tLoading model parameter values ...')
    with open(parameters_filename, 'rb') as fd:
        lasagne.layers.set_all_param_values(network, cPickle.load(fd))

    if updates_filename:
        logging.info('\tLoading optimization training updates ...')
        with open(updates_filename, 'rb') as fd:
            for p, value in zip(updates.keys(), cPickle.load(fd)):
                p.set_value(value)


def save_model(model, updates, destination_directory, epoch):
    with open(os.sep.join([destination_directory, 'parameters_epoch-{}.pkl'.format(epoch) ]), 'wb') as fd:
        cPickle.dump(lasagne.layers.get_all_param_values(model), fd, protocol=cPickle.HIGHEST_PROTOCOL)
    with open(os.sep.join([destination_directory, 'updates_epoch-{}.pkl'.format(epoch) ]), 'wb') as fd:
        cPickle.dump( [ p.get_value() for p in updates.keys() ], fd, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    config_params = vars(args)
    logging.info('Configuration parameters:\n' + pprint.pformat(config_params))

    if len(config_params['filter_size_conv_layers'])!= 2 or \
       len(config_params['nr_of_filters_conv_layers'])!=2 or \
       len(config_params['activations'])!=2 or \
       len(config_params['L2'])!=4 :
        raise Exception('Check if the input --filter_size_conv_layers, --nr_of_filters_conv_layers and --activations are lists of size 2, and the --L2 field needs a value list of 4 values.')

    logging.info('Creating output directory ...')
    if os.path.exists(config_params['destination_directory']):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(config_params['destination_directory'])
    with open(os.sep.join([config_params['destination_directory'], 'configs.pkl']), 'wb') as fd:
        cPickle.dump(config_params, fd, protocol=cPickle.HIGHEST_PROTOCOL)

    logging.info('Loading and preparing data ...')
    logging.debug('\tLoading training data ...')
    train_data = Dataset.Dataset(os.sep.join([config_params['source_dir_data'], 'training_batches_size-' + str(config_params['batch_size'])]),
                                 memory = config_params['memory'],
                                 x_filename = os.sep.join([config_params['source_dir_data'], 'training.csv']),
                                 y_filename = os.sep.join([config_params['source_dir_data'], 'training_lbl.csv']),
                                 batch_size = config_params['batch_size'],
                                )
    logging.debug('\tLoading validation data ...')
    valid_data = Dataset.Dataset(os.sep.join([config_params['source_dir_data'], 'valid_batches_size-' + str(config_params['batch_size'])]),
                                 memory = config_params['memory'],
                                 x_filename = os.sep.join([config_params['source_dir_data'], 'valid.csv']),
                                 y_filename = os.sep.join([config_params['source_dir_data'], 'valid_lbl.csv']),
                                 batch_size = config_params['batch_size'],
                                )
    logging.debug('\tLoading test data ...')
    test_data = Dataset.Dataset(os.sep.join([config_params['source_dir_data'], 'test_batches_size-' + str(config_params['batch_size'])]),
                                memory = config_params['memory'],
                                x_filename = os.sep.join([config_params['source_dir_data'], 'test.csv']),
                                y_filename = os.sep.join([config_params['source_dir_data'], 'test_lbl.csv']),
                                batch_size = config_params['batch_size'],
                               )

    logging.info('Building the model ...')

    # allocate symbolic variables for the data
    x_batch = T.imatrix('x')
    y_batch = T.ivector('y')

    # define/load the network
    output_layer = networks.buildDCNNPaper(batch_size=config_params['batch_size'],
                                           vocab_size=config_params['vocab_size'],
                                           embeddings_size=config_params['word_vector_size'],
                                           filter_sizes=config_params['filter_size_conv_layers'],
                                           nr_of_filters=config_params['nr_of_filters_conv_layers'],
                                           activations=config_params['activations'],
                                           ktop=config_params['ktop'],
                                           dropout=config_params['dropout_value'],
                                           output_classes=config_params['output_classes'],
                                           padding='last'
                                          )

    # Kalchbrenner uses a fine-grained L2 regularization in the Matlab code, default values taken from Matlab code
    # Training objective
    l2_layers = []
    for layer in lasagne.layers.get_all_layers(output_layer):
        if isinstance(layer,(DCNN.embeddings.SentenceEmbeddingLayer,DCNN.convolutions.Conv1DLayerSplitted,lasagne.layers.DenseLayer)):
            l2_layers.append(layer)
    loss_train = lasagne.objectives.aggregate(
                                              lasagne.objectives.categorical_crossentropy(
                                                                                          lasagne.layers.get_output(output_layer, x_batch),
                                                                                          y_batch
                                                                                         ),
                                              mode='mean'
                                             ) + lasagne.regularization.regularize_layer_params_weighted(
                                                                                                         dict(zip(l2_layers, config_params['L2'])),
                                                                                                         lasagne.regularization.l2
                                                                                                        )

    # In the matlab code, Kalchbrenner works with a adagrad reset mechanism, if the para --adagrad_reset has value 0, no reset will be applied
    all_params = lasagne.layers.get_all_params(output_layer)
    updates, accumulated_grads = utils.adagrad(loss_train, all_params, config_params['learning_rate'])
    #updates = lasagne.updates.adagrad(loss_train, all_params, config_params['learning_rate'])

    if config_params['parameters_filename']:
        load_model(output_layer, config_params['parameters_filename'], updates, config_params['updates_filename'])


    # validating/testing
    pred = T.argmax(lasagne.layers.get_output(output_layer, x_batch, deterministic=True),axis=1)
    correct_predictions = T.eq(pred, y_batch)

    train_model = theano.function(inputs=[x_batch,y_batch], outputs=loss_train, updates=updates)
    valid_model = theano.function(inputs=[x_batch,y_batch], outputs=correct_predictions)
    test_model = theano.function(inputs=[x_batch], outputs=pred)

    logging.info('Starting training ...')
    #theano.config.exception_verbosity = 'high'


    # assemble test ground truth once to save along with the predictions at each better model find
    test_ground_truth = { 'ground_truth': [], 'tweet_ids': [], 'user_ids': [] }
    for minibatch_index in range(test_data.num_batches()):
        batch = test_data.get_batch(minibatch_index)
        test_ground_truth['ground_truth'].append(batch['y'])
        test_ground_truth['tweet_ids'].append(batch['tweet_id'])
        test_ground_truth['user_ids'].append(batch['user_id'])
    test_ground_truth['ground_truth'] = numpy.concatenate(test_ground_truth['ground_truth'])[:test_data.num_samples()]
    test_ground_truth['tweet_ids'] = numpy.concatenate(test_ground_truth['tweet_ids'])[:test_data.num_samples()]
    test_ground_truth['user_ids'] = numpy.concatenate(test_ground_truth['user_ids'])[:test_data.num_samples()]

    best_validation_accuracy = 0
    batch_size = config_params['batch_size']
    num_batches_per_valid = train_data.num_batches()
    if config_params['valid_freq'] >= 1.0 and config_params['valid_freq'] < train_data.num_batches():
        num_batches_per_valid = int(config_params['valid_freq'])
    elif config_params['valid_freq'] > 0.0 and config_params['valid_freq'] < 1.0:
        num_batches_per_valid = int(config_params['valid_freq']*train_data.num_batches())
    start = time.time()
    for epoch in range(1, config_params['n_epochs']+1):
        logging.info('Starting epoch {}/{} ...'.format(epoch, config_params['n_epochs']))
        permutation = numpy.random.permutation(train_data.num_batches())
        batch_counter = 0
        train_loss = 0
        for minibatch_index in permutation:
            sys.stdout.write('{}/{} batches processed\r'.format(batch_counter, train_data.num_batches()))
            batch = train_data.get_batch(minibatch_index)
            train_loss += train_model(batch['x'], batch['y'])
            batch_counter += 1

            if batch_counter % num_batches_per_valid == 0:
                logging.debug('\tCalculating validation accuracy at batch {}/{} of epoch {}/{} ...'.format(batch_counter, train_data.num_batches(), epoch, config_params['n_epochs']))
                accuracies = []
                for minibatch_index in range(valid_data.num_batches()):
                    batch = valid_data.get_batch(minibatch_index)
                    accuracies.append(valid_model(batch['x'], batch['y']))
                valid_accuracy = float(numpy.concatenate(accuracies)[:valid_data.num_samples()].sum())/valid_data.num_samples()*100
                if valid_accuracy > best_validation_accuracy:
                    logging.info('\t\tBetter validation accuracy found after batch {}/{} of epoch {}/{}: {} %'.format(batch_counter, train_data.num_batches(), epoch, config_params['n_epochs'], valid_accuracy))
                    best_validation_accuracy = valid_accuracy
                    logging.debug('\t\tCalculating test accuracy ...')
                    predictions = []
                    for minibatch_index in range(test_data.num_batches()):
                        batch = test_data.get_batch(minibatch_index)
                        predictions.append(test_model(batch['x']))
                    predictions = numpy.concatenate(predictions)[:test_data.num_samples()]
                    test_accuracy = float(numpy.equal(predictions, test_ground_truth['ground_truth']).sum())/test_data.num_samples()*100
                    logging.info('\t\tTest accuracy over the best validation model found after batch {}/{} of epoch {}/{}: {} %'.format(batch_counter, train_data.num_batches(), epoch, config_params['n_epochs'], test_accuracy))
                    logging.info('Saving model and predictions ...')
                    save_model(output_layer, updates, config_params['destination_directory'], epoch)
                    test_ground_truth['predictions'] = predictions
                    with open(os.sep.join([config_params['destination_directory'], 'predictions_epoch-{}.pkl'.format(epoch) ]), mode='wt',) as fd:
                        cPickle.dump(test_ground_truth, fd, protocol=cPickle.HIGHEST_PROTOCOL)
                    del test_ground_truth['predictions']

        logging.debug('\tCalculating training accuracy ...')
        accuracies = []
        for minibatch_index in range(train_data.num_batches()):
            batch = train_data.get_batch(minibatch_index)
            accuracies.append(valid_model(batch['x'], batch['y']))
        train_accuracy = float(numpy.concatenate(accuracies)[:train_data.num_samples()].sum())/train_data.num_samples()*100

        train_loss /= float(train_data.num_batches())
        logging.info('Epoch {} results:'.format(epoch))
        logging.info('\tTraining loss: {}'.format(train_loss))
        logging.info('\tTraining accuracy: {} %'.format(train_accuracy))
        logging.info('\tValidation accuracy: {} %'.format(valid_accuracy))
        logging.info('\tBest validation accuracy: {} %'.format(best_validation_accuracy))
        logging.info('\tTest accuracy over the best validation model: {} %'.format(test_accuracy))
        logging.info('\tCSV format (epoch, training_loss, training accuracy, validation accuracy): {}, {}, {}, {}'.format(epoch, train_loss, train_accuracy, valid_accuracy))

        if config_params['adagrad_reset'] > 0:
            if epoch % config_params['adagrad_reset'] == 0:
                utils.reset_grads(accumulated_grads)

        if config_params['max_minutes'] > 0 and (time.time() - start) > (config_params['max_minutes']*60):
            logging.info('Stopping training due to time limit ...')
            break

    logging.info('Finished.')
