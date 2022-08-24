#!/usr/bin/env python
# Python 2 due to Theano/Lasagne

"""
"""

import argparse
import logging
import os
import sys
import cPickle
import pprint
import Dataset
import theano
import theano.tensor as T
import lasagne
import numpy
import json

sys.path.append(os.sep.join(['.', 'DynamicCNN']))
import networks
import DCNN


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    # input/output
    parser.add_argument('--batches_directory', '-b',
                        required=True,
                        help='Directory where the batches will be generated or read from.')
    parser.add_argument('--data_filename', '-x',
                        help='Filename containing the input data to be classified if --batches_directory doesn\'t exist (in CSV format).')
    parser.add_argument('--label_filename', '-y',
                        help='Filename containing the input data label if --batches_directory doesn\'t exist (in CSV format).')
    parser.add_argument('--destination_directory', '-e',
                        required=True,
                        help='Directory where the outputs will be stored.')
    parser.add_argument('--configuration_filename', '-c',
                        required=True,
                        help='Filename containing the model configuration in Pickle format.')
    parser.add_argument('--parameters_filename', '-p',
                        required=True,
                        help='Filename containing the model parameters in Pickle format.')
    parser.add_argument('--memory', '-m',
                        action='store_true',
                        default=False,
                        help='If specified, batches are fully stored in memory. Default = False.')
    parser.add_argument('--generate_features', '-f',
                        action='store_true',
                        default=False,
                        help='Generate intermediate features along classification data.')
    parser.add_argument('--features_libsvm', '-l',
                        action='store_true',
                        default=False,
                        help='If true, features are generated in a separate file in LIBSVM format along with the label.')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information.')

    args = parser.parse_args()
    if not os.path.exists(args.batches_directory) and (args.data_filename == None or args.label_filename == None):
        parser.error('If --batches_directory doesn\'t exist, --data_filename and --label_filename must be specified.')
    return args


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    logging.info('Setting configuration parameters...')
    config_params = vars(args)
    with open(config_params['configuration_filename'], mode='rb') as fd:
        for key, value in cPickle.load(fd).items():
            if key not in config_params:
                config_params[key] = value
    logging.info('Configuration parameters:\n' + pprint.pformat(config_params))

    logging.info('Creating output directory ...')
    if os.path.exists(config_params['destination_directory']):
        logging.error('Destination directory already exists. Quitting ...')
        sys.exit(1)
    os.makedirs(config_params['destination_directory'])
    with open(os.sep.join([config_params['destination_directory'], 'configs.pkl']), 'wb') as fd:
        cPickle.dump(config_params, fd, protocol=cPickle.HIGHEST_PROTOCOL)

    logging.info('Loading and preparing data ...')
    data = Dataset.Dataset(config_params['batches_directory'],
                           memory = config_params['memory'],
                           x_filename = config_params['data_filename'],
                           y_filename = config_params['label_filename'],
                           batch_size = config_params['batch_size'],
                          )

    logging.info('Building the model ...')
    # allocate symbolic variables for the data
    x_batch = T.imatrix('x')
    y_batch = T.ivector('y')
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
    logging.debug('\tLoading model parameter values ...')
    with open(config_params['parameters_filename'], 'rb') as fd:
        lasagne.layers.set_all_param_values(output_layer, cPickle.load(fd))

    if config_params['generate_features']:
        feature_vector_layer, prediction_layer = lasagne.layers.get_output([output_layer.input_layer.input_layer, output_layer], x_batch, deterministic=True)
        feature_vectors_model = theano.function(inputs=[x_batch], outputs=feature_vector_layer)
        if config_params['features_libsvm']:
            libsvm_fd = open(os.sep.join([config_params['destination_directory'], 'cnn_features.libsvm']), mode='wt')
    else:
        prediction_layer = lasagne.layers.get_output(output_layer, x_batch, deterministic=True)
    prediction_model = theano.function(inputs=[x_batch], outputs=T.argmax(prediction_layer, axis=1))

    logging.info('Classifying data ...')
    output = []
    accuracies = []
    for minibatch_index in range(data.num_batches()):
        if config_params['debug']:
            sys.stdout.write('{}/{} batches processed\r'.format(minibatch_index, data.num_batches()))
        batch = data.get_batch(minibatch_index)
        if config_params['generate_features']:
            feats = feature_vectors_model(batch['x'])
        preds = prediction_model(batch['x'])
        accuracies.append(numpy.equal(batch['y'], preds))
        for idx in range(config_params['batch_size']):
            output.append({'tweet_id': batch['tweet_id'][idx],
                           'author_id': batch['user_id'][idx],
                           'x': batch['x'][idx].tolist(),
                           'y': int(batch['y'][idx]),
                           'prediction': int(preds[idx]),
                          })
            if config_params['generate_features']:
                if config_params['features_libsvm']:
                    row = [ str(batch['y'][idx]) ]
                    for index, value in enumerate(feats[idx].ravel(), 1):
                        row.append('{}:{}'.format(index, value))
                    libsvm_fd.write(' '.join(row) + '\n')
                else:
                    output[-1]['cnn_features'] = feats[idx].ravel().tolist()

    if config_params['generate_features'] and config_params['features_libsvm']:
        libsvm_fd.close()
    documentation_dict = {'tweet_id': '<integer> Unique tweet ID defined by Twitter.',
                          'author_id': '<integer> Unique user ID defined by Twitter.',
                          'x': '<variable-length integer list> Input vector fed to the network. Represent Unicode characters mapped to integer values.',
                          'y': '<integer> Label ID.',
                          'cnn_features': '<fixed-length float list> Features generated by the layer just before the classification layer (softmax). Optional.',
                          'prediction': '<integer> Label ID as predicted by the network.',
                         }
    with open(os.sep.join([config_params['destination_directory'], 'classification.json']), mode='wt') as fd:
        json.dump({'data_description': documentation_dict, 'data_values': output}, fd, sort_keys=True, ensure_ascii=True)
    logging.info('Final accuracy: {}'.format( float(numpy.concatenate(accuracies).sum())/data.num_samples()*100 ))

    logging.info('Finished.')
