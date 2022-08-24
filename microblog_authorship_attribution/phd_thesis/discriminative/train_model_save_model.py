#!/usr/bin/env python
# Python 2 due to Theano/Lasagne

"""
"""

import argparse
import logging
import os
import sys
import pprint
import numpy
import theano
import theano.tensor as T
import lasagne
import networks
import DCNN
import utils
import time
import cPickle


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    # input/output
    parser.add_argument('--source_dir_data', '-a',
                        dest='source_dir_data',
                        required=True,
                        help='Directory where the input data files (in CSV format) are stored.')
    parser.add_argument('--destination_directory', '-b',
                        dest='destination_directory',
                        required=True,
                        help='Directory where the outputs will be stored.')
    parser.add_argument('--output_classes',
                        required=True,
                        type=int,
                        help='Number of output classes.')
    parser.add_argument('--vocab_size',
                        required=True,
                        type=int,
                        help='Vocabulary size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=40,
                        help='Batch size. Default = 40.')                                       #changed from original
    parser.add_argument('--model_filename', '-m',
                        dest='model_filename',
                        help='Filename containing the model (network+parameters) in Pickle format.')
    parser.add_argument('--updates_filename', '-u',
                        dest='updates_filename',
                        help='Filename containing the training updates in Pickle format.')
    # training settings
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
    #parser.add_argument('--valid_freq',
    #                    type=int,
    #                    default=500,
    #                    help='Number of batches processed until we validate. Default = 500.')   # changed from original
    parser.add_argument('--adagrad_reset',
                        type=int,
                        default=5,
                        help='Resets the adagrad cumulative gradient after x epochs. If the value is 0, no reset will be executed. Default = 5.')
    # network paras
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
    return parser.parse_args()


def read_pad_and_sort_data(x_file, y_file, padding_value=None):
    sorted_dict = {}
    x_data = []
    i = 0
    # TODO - to evolve this block reading the file as a CSV of ints. Maybe Pandas can do this in a couple of lines and faster
    with open(x_file, mode='rt') as fd:
        lines = fd.readlines()[1:]      # avoid header
    for line in lines:
        words = line.split(',')[1:]     # avoid tweet id
        result = []
        for word in words:
            word_i = int(word)
            result.append(word_i)
        x_data.append(result)
        length=len(result)
        if length in sorted_dict:
            sorted_dict[length].append(i)
        else:
            sorted_dict[length]=[i]
        i += 1

    # pad
    max_length = max(sorted_dict.keys())
    for data in x_data:
        pad_length = max_length - len(data)
        if padding_value == None:   # pad with the same array data - circular data
            data += ( data * (max_length // len(data)) )[:pad_length]
        else:
            data += [padding_value] * pad_length

    with open(y_file, mode='rt') as fd:
        lines = fd.readlines()[1:]      # avoid header
    y_data = []
    for line in lines:
        y_data.append(int(line.split(',')[1]))

    new_train_list = []
    new_label_list = []
    lengths = []
    for length, indexes in sorted(sorted_dict.items(), key=lambda x: x[0] ):
        for index in indexes:
            new_train_list.append(x_data[index])
            new_label_list.append(y_data[index])
            lengths.append(length)

    return numpy.asarray(new_train_list, dtype=numpy.int32), numpy.asarray(new_label_list, dtype=numpy.int32), lengths


def pad_to_batch_size(array, batch_size, zero_padding=True):
    rows_extra = batch_size - (array.shape[0] % batch_size)
    if len(array.shape) == 1:
        if zero_padding:
            padding = numpy.zeros((rows_extra,),dtype=numpy.int32)
        else:   # repeat the last rows
            padding = array[-rows_extra:]
        return numpy.concatenate((array, padding))
    else:
        if zero_padding:
            padding = numpy.zeros((rows_extra,array.shape[1]), dtype=numpy.int32)
        else:   # repeat the last rows
            padding = array[-rows_extra:,:]
        return numpy.vstack((array, padding))


def extend_lengths(length_list, batch_size):
    elements_extra = batch_size - (len(length_list) % batch_size)
    length_list.extend([length_list[-1]] * elements_extra)


def save_model(model, updates, destination_directory, epoch):
    with open(os.sep.join([destination_directory, 'model_epoch-{}.pkl'.format(epoch) ]), 'wb') as fd:
        cPickle.dump(model, fd, protocol=cPickle.HIGHEST_PROTOCOL)
    with open(os.sep.join([destination_directory, 'updates_epoch-{}.pkl'.format(epoch) ]), 'wb') as fd:
        cPickle.dump( [ p.get_value() for p in updates.keys() ], fd, protocol=cPickle.HIGHEST_PROTOCOL)


def load_model(model_filename):
    logging.info('\tLoading model ...')
    with open(model_filename, 'rb') as fd:
        return cPickle.load(fd)


def load_updates(updates_filename, updates):
    logging.info('\tLoading optimization training updates ...')
    with open(updates_filename, 'rb') as fd:
        for p, value in zip(updates.keys(), cPickle.load(fd)):
            p.set_value(value)


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

    logging.info('Creating output directory ...')
    if os.path.exists(args.destination_directory):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(args.destination_directory)

    logging.info('Starting classification ...')
    hyperparas = vars(args)
    logging.info('Parameters:\n' + pprint.pformat(hyperparas))

    if len(hyperparas['filter_size_conv_layers'])!= 2 or \
       len(hyperparas['nr_of_filters_conv_layers'])!=2 or \
       len(hyperparas['activations'])!=2 or \
       len(hyperparas['L2'])!=4 :
        raise Exception('Check if the input --filter_size_conv_layers, --nr_of_filters_conv_layers and --activations are lists of size 2, and the --L2 field needs a value list of 4 values.')

    with open(os.sep.join([args.destination_directory, 'model_config.pkl']), 'wb') as fd:
        cPickle.dump(hyperparas, fd, protocol=cPickle.HIGHEST_PROTOCOL)


    #######################
    # LOAD  TRAINING DATA #
    #######################
    logging.info('Loading the training data ...')

    # we order the input according to length and pad all sentences until the maximum length
    # at training time however, we will use the 'length' array to shrink that matrix following the largest sentence within a batch
    # in practice, this means that batches are padded with 1 or 2 elements, or aren't even padded at all.
    train_x_indexes, train_y, train_lengths = read_pad_and_sort_data(os.sep.join([args.source_dir_data, 'training.csv']), os.sep.join([args.source_dir_data, 'training_lbl.csv']))
    dev_x_indexes, dev_y, dev_lengths = read_pad_and_sort_data(os.sep.join([args.source_dir_data, 'valid.csv']), os.sep.join([args.source_dir_data, 'valid_lbl.csv']))
    test_x_indexes, test_y, test_lengths = read_pad_and_sort_data(os.sep.join([args.source_dir_data, 'test.csv']), os.sep.join([args.source_dir_data, 'test_lbl.csv']))

    # train data
    n_train_batches = len(train_lengths) // hyperparas['batch_size']

    # dev data
    # to be able to do a correct evaluation, we pad a number of rows to get a multiple of the batch size
    dev_x_indexes_extended = pad_to_batch_size(dev_x_indexes, hyperparas['batch_size'], zero_padding=False)
    dev_y_extended = pad_to_batch_size(dev_y, hyperparas['batch_size'], zero_padding=False)
    n_dev_batches = dev_x_indexes_extended.shape[0] // hyperparas['batch_size']
    n_dev_samples = len(dev_y)
    extend_lengths(dev_lengths, hyperparas['batch_size'])

    # test data
    test_x_indexes_extended = pad_to_batch_size(test_x_indexes, hyperparas['batch_size'], zero_padding=False)
    test_y_extended = pad_to_batch_size(test_y, hyperparas['batch_size'], zero_padding=False)
    n_test_batches = test_x_indexes_extended.shape[0] // hyperparas['batch_size']
    n_test_samples = len(test_y)
    extend_lengths(test_lengths, hyperparas['batch_size'])


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    logging.info('Building the model ...')

    # allocate symbolic variables for the data
    X_batch = T.imatrix('x')
    y_batch = T.ivector('y')

    # define/load the network
    if not args.model_filename:
        output_layer = networks.buildDCNNPaper(batch_size=hyperparas['batch_size'],
                                               vocab_size=hyperparas['vocab_size'],
                                               embeddings_size=hyperparas['word_vector_size'],
                                               filter_sizes=hyperparas['filter_size_conv_layers'],
                                               nr_of_filters=hyperparas['nr_of_filters_conv_layers'],
                                               activations=hyperparas['activations'],
                                               ktop=hyperparas['ktop'],
                                               dropout=hyperparas['dropout_value'],
                                               output_classes=hyperparas['output_classes'],
                                               padding='last'
                                              )
    else:
        output_layer = load_model(args.model_filename)

    # Kalchbrenner uses a fine-grained L2 regularization in the Matlab code, default values taken from Matlab code
    # Training objective
    l2_layers = []
    for layer in lasagne.layers.get_all_layers(output_layer):
        if isinstance(layer,(DCNN.embeddings.SentenceEmbeddingLayer,DCNN.convolutions.Conv1DLayerSplitted,lasagne.layers.DenseLayer)):
            l2_layers.append(layer)
    loss_train = lasagne.objectives.aggregate(
                                              lasagne.objectives.categorical_crossentropy(
                                                                                          lasagne.layers.get_output(output_layer, X_batch),
                                                                                          y_batch
                                                                                         ),
                                              mode='mean'
                                             ) + lasagne.regularization.regularize_layer_params_weighted(
                                                                                                         dict(zip(l2_layers, hyperparas['L2'])),
                                                                                                         lasagne.regularization.l2
                                                                                                        )

    # validating/testing
    #loss_eval = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(output_layer,X_batch,deterministic=True),y_batch)
    pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True),axis=1)
    correct_predictions = T.eq(pred, y_batch)

    # In the matlab code, Kalchbrenner works with a adagrad reset mechanism, if the para --adagrad_reset has value 0, no reset will be applied
    all_params = lasagne.layers.get_all_params(output_layer)
    updates, accumulated_grads = utils.adagrad(loss_train, all_params, hyperparas['learning_rate'])
    #updates = lasagne.updates.adagrad(loss_train, all_params, hyperparas['learning_rate'])
    if args.updates_filename:
        load_updates(args.updates_filename, updates)

    train_model = theano.function(inputs=[X_batch,y_batch], outputs=loss_train, updates=updates)
    valid_model = theano.function(inputs=[X_batch,y_batch], outputs=correct_predictions)
    test_model = theano.function(inputs=[X_batch,y_batch], outputs=correct_predictions)


    ###############
    # TRAIN MODEL #
    ###############
    logging.info('Starting training ...')
    #theano.config.exception_verbosity = 'high'

    best_validation_accuracy = 0
    batch_size = hyperparas['batch_size']
    start = time.time()
    for epoch in range(1, hyperparas['n_epochs']+1):
        logging.info('Starting epoch {}/{} ...'.format(epoch, hyperparas['n_epochs']))
        permutation = numpy.random.permutation(n_train_batches)
        batch_counter = 0
        train_loss = 0
        for minibatch_index in permutation:
            sys.stdout.write('%d/%d batches processed\r' % (batch_counter, n_train_batches))
            x_input = train_x_indexes[
                                      minibatch_index*batch_size : (minibatch_index+1)*batch_size,
                                      0 : train_lengths[(minibatch_index+1)*batch_size - 1 ]
                                     ]
            y_input = train_y[ minibatch_index*batch_size : (minibatch_index+1)*batch_size ]
            train_loss += train_model(x_input, y_input)
            batch_counter += 1

        logging.debug('\tCalculating training accuracy ...')
        accuracy_train = []
        for minibatch_index in range(n_train_batches):
            x_input = train_x_indexes[
                                      minibatch_index*batch_size : (minibatch_index+1)*batch_size,
                                      0 : train_lengths[ (minibatch_index+1)*batch_size - 1 ]
                                     ]
            y_input = train_y[ minibatch_index*batch_size : (minibatch_index+1)*batch_size ]
            accuracy_train.append(valid_model(x_input, y_input))
        training_accuracy = numpy.concatenate(accuracy_train).sum()/float(n_train_batches*batch_size)*100

        logging.debug('\tCalculating validation accuracy ...')
        accuracy_valid=[]
        for minibatch_index in range(n_dev_batches):
            x_input = dev_x_indexes_extended[
                                             minibatch_index*batch_size : (minibatch_index+1)*batch_size,
                                             0 : dev_lengths[(minibatch_index+1)*batch_size - 1 ]
                                            ]
            y_input = dev_y_extended[ minibatch_index*batch_size : (minibatch_index+1)*batch_size ]
            accuracy_valid.append(valid_model(x_input, y_input))
        validation_accuracy = numpy.concatenate(accuracy_valid)[0:n_dev_samples].sum()/float(n_dev_samples)*100
        if validation_accuracy > best_validation_accuracy:
            logging.debug('\t\tBetter validation accuracy found at epoch {}/{}: {} %'.format(epoch, hyperparas['n_epochs'], validation_accuracy))
            best_validation_accuracy = validation_accuracy
            logging.debug('\t\tCalculating test accuracy ...')
            accuracy_test=[]
            for minibatch_index in range(n_test_batches):
                x_input = test_x_indexes_extended[
                                                  minibatch_index*batch_size : (minibatch_index+1)*batch_size,
                                                  0 : test_lengths[(minibatch_index+1)*batch_size - 1 ]
                                                 ]
                y_input = test_y_extended[ minibatch_index*batch_size : (minibatch_index+1)*batch_size ]
                accuracy_test.append(test_model(x_input, y_input))
            test_accuracy = numpy.concatenate(accuracy_test)[0:n_test_samples].sum()/float(n_test_samples)*100
            logging.info('Saving models ...')
            save_model(output_layer, updates, args.destination_directory, epoch)

        train_loss /= float(n_train_batches)
        logging.info('Epoch {} results:'.format(epoch))
        logging.info('\tTraining loss: {}'.format(train_loss))
        logging.info('\tTraining accuracy: {} %'.format(training_accuracy))
        logging.info('\tValidation accuracy: {} %'.format(validation_accuracy))
        logging.info('\tBest validation accuracy: {} %'.format(best_validation_accuracy))
        logging.info('\tTest accuracy over the best validation model: {} %'.format(test_accuracy))
        logging.info('\tCSV format (epoch, training_loss, training accuracy, validation accuracy): {}, {}, {}, {}'.format(epoch, train_loss, training_accuracy, validation_accuracy))

        if hyperparas['adagrad_reset'] > 0:
            if epoch % hyperparas['adagrad_reset'] == 0:
                utils.reset_grads(accumulated_grads)

        if hyperparas['max_minutes'] > 0 and (time.time() - start) > (hyperparas['max_minutes']*60):
            logging.info('Stopping training due to time limit ...')
            break

    logging.info('Finished.')
