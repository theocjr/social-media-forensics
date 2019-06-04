"""
"""

import logging
import os
import sys
import json
import math
import random
import copy
import numpy


class Dataset():

    _logger = None
    _data_dir = None
    _num_samples = None
    _num_batches = None
    _memory_data = None


    def __init__(self, batches_destination_dir, memory=False, x_filename = None, y_filename = None, batch_size = None, padding_value=None):
        self._logger = logging.getLogger(__name__)
        self._data_dir = batches_destination_dir
        if os.path.exists(self._data_dir):
            self._logger.debug('Found batches data directory at {}. Skipping batches generation ...'.format(self._data_dir))
            with open(os.sep.join([self._data_dir, 'data_description.json']), mode='rt') as fd:
                data_description = json.load(fd)
        else:
            if x_filename == None or y_filename == None:
                raise Exception('x_filename or y_filename not specified to generate batches.')
            os.makedirs(self._data_dir)
            data_description = self._generate_minibatches(x_filename, y_filename, batch_size, padding_value)
        self._num_samples = data_description['num_samples']
        self._num_batches = data_description['num_batches']

        if memory:
            self._logger.debug('Loading dataset into memory ...')
            self._memory_data = {}
            for batch_id in range(self._num_batches):
                with open(os.sep.join([self._data_dir, str(batch_id) + '.json']), mode='rt') as fd:
                    batch = json.load(fd)
                batch['x'] = numpy.asarray(batch['x'], dtype=numpy.int32)
                batch['y'] = numpy.asarray(batch['y'], dtype=numpy.int32)
                self._memory_data[batch_id] = batch


    def get_batch(self, batch_id):
        if self._memory_data:
            return self._memory_data[batch_id]
        else:
            with open(os.sep.join([self._data_dir, str(batch_id) + '.json']), mode='rt') as fd:
                batch = json.load(fd)
            batch['x'] = numpy.asarray(batch['x'], dtype=numpy.int32)
            batch['y'] = numpy.asarray(batch['y'], dtype=numpy.int32)
            return batch


    def num_samples(self):
        return self._num_samples


    def num_batches(self):
        return self._num_batches


    def _generate_minibatches(self, x_filename, y_filename, batch_size, padding_value):
        '''The input is ordered according to length and partitioned into
        minibatches. The sentences are padded until the maximum length inside the
        batch. In practice, this means that sentences are padded with 1 or 2
        elements, or aren't even padded at all.
        '''
        self._logger.debug('Reading data to sort by length ...')
        lengths = {}
        idx = 0
        with open(x_filename, mode='rt') as fd:
            fd.readline()    # header
            for line in fd:
                length = len(line.split(',')) - 1   # ignore first column (tweet ID)
                if length in lengths:
                    lengths[length].append(idx)
                else:
                    lengths[length] = [ idx ]
                idx += 1
        num_samples = idx
        num_batches = int(math.ceil(float(idx) / batch_size))       # the float cast is just to avoid an off-by-one error if Python 2 was used
        for _, idxs in lengths.items():             # try to avoid batches with few different users
            random.shuffle(idxs)

        self._logger.debug('Defining batches ...')
        line_batch_map = {}
        current_batch = 0
        current_batch_size = 0
        for _, idxs in sorted( lengths.items(), key=lambda x: x[0] ):
            for idx in idxs:
                if current_batch_size == batch_size:
                    current_batch += 1
                    current_batch_size = 0
                line_batch_map[idx] = current_batch
                current_batch_size += 1

        self._logger.debug('Reading data to assemble batches ...')
        batches = {}
        with open(x_filename, mode='rt') as x_fd, open(y_filename, mode='rt') as y_fd:
            x_fd.readline()    # header
            y_fd.readline()    # header
            idx = 0
            batch_counter = 0   # processing feedback
            for x_line in x_fd:
                y_line = y_fd.readline()
                batch_id = line_batch_map[idx]
                if batch_id not in batches:
                    batches[batch_id] = {'x': [], 'y': [], 'tweet_id': [], 'user_id':[]}
                x_fields = list(map(int, x_line.split(',')))
                y_fields = list(map(int, y_line.split(',')))
                batches[batch_id]['x'].append(x_fields[1:])
                batches[batch_id]['y'].append(y_fields[1])
                batches[batch_id]['tweet_id'].append(x_fields[0])
                batches[batch_id]['user_id'].append(y_fields[0])
                if len(batches[batch_id]['x']) == batch_size:
                    self._pad_storage_batch(batch_id, batches[batch_id], padding_value)
                    del batches[batch_id]
                    if self._logger.getEffectiveLevel() == logging.DEBUG:
                        batch_counter += 1
                        sys.stdout.write('{}/{} batches processed\r'.format(batch_counter, num_batches))
                idx += 1

        if len(batches.keys()) > 1:
            raise Exception('Error generating mini-batches. Only the last mini-batch was expected to have less than batch_size elements.')

        # repeat samples in last mini-batch until batch_size
        for batch_id in batches:
            batch = batches[batch_id]
            original_length = len(batch['x'])
            samples_remaining = batch_size - original_length
            for idx in range(samples_remaining):
                batch['x'].append( copy.deepcopy(batch['x'][idx % original_length]) )
                batch['y'].append(batch['y'][idx % original_length])
                batch['tweet_id'].append(batch['tweet_id'][idx % original_length])
                batch['user_id'].append(batch['user_id'][idx % original_length])
            self._pad_storage_batch(batch_id, batch, padding_value)

        data_description = {'num_samples': num_samples,
                            'batch_size': batch_size,
                            'num_batches': num_batches,
                           }
        with open(os.sep.join([self._data_dir, 'data_description.json']), mode='wt') as fd:
            json.dump(data_description, fd, sort_keys=True, ensure_ascii=True)

        return data_description


    def _pad_storage_batch(self, batch_id, batch, padding_value):
        batch['pad_length'] = []
        max_length = max([ len(sample) for sample in batch['x'] ])
        for sample in batch['x']:
            pad_length = max_length - len(sample)
            if padding_value:
                sample += [ padding_value ] * pad_length
            else:   # pad with the same array data - circular data
                sample += ( sample * (max_length // len(sample)) )[:pad_length]
            batch['pad_length'].append(pad_length)
        with open(os.sep.join([self._data_dir, str(batch_id) + '.json']), mode='wt') as fd:
            json.dump(batch, fd, sort_keys=True, ensure_ascii=True)
