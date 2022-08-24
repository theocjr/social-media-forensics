#!/usr/bin/env python
# coding: utf-8

# # Pipeline
# 
# 1. **Pre-processing (done previously):** Pre-process author data with the following configuration:
#     1. Keep hashtags
#     1. Tag numbers, urls, and mentions
# 1. **Language Model (LM) Training (done previously):** Fine-tune one language model per author
# 1. **Prediction Strategy: (done by this code)**
#     1. Mask some tokens (one at a time)
#     1. Calculate the probability for the true token using each author LM for each masked token. The result is a multi-dimensional array (P, A) for each message, where
#         * P: Number of masked tokens on a message (variable, eventually empty)
#         * A: Number of authors
#     1. Transform the probabilities over a vocabulary into a probabilities over the authors (normalizing the probilities in the dimension A)
#     1. Order the author probabilities (dimension A)
#     1. For each message, sums the order an author appears for each masked token of that message. The author with the highest value is the predicted author for that message.


import torch
import argparse
import logging
import os
import sys
import json
import pprint
import collections
import transformers
import glob
import numpy
import joblib


# ## Global Variables

"""
Ordered list of authors ID. The index of each author in this list is used to identify
an author in this experiment.
"""
authors_ids = None

"""
List of lists.
The elements of the outer list are the set of messages (HuggingFace dataset) of an author (in the same order of `authors_ids` list).
The elements of the inner lists are dictionaries representing one single message (in the order they appear in the stored file), containing:
    - Tweet ID (integer)
    - text (string)
    - tokenized text (list)
    - indexes to mask in the tokenized text list (list)
    - predictions (multidimensional numpy array):
            Dimension P: Number of tokens on a message (variable)
            Dimension A: Number of authors

"""
authors_messages = None

"""
Tokenizer.
"""
tokenizer = None


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument('--tweets_dir',
                        required=True,
                        help='Directory cointaining the tweets to be evaluated (one sub-directory per author).')
    parser.add_argument('--models_dir',
                        required=True,
                        help='Directory cointaining the language models to be used in evaluation (one sub-directory per author).')
    parser.add_argument('--output_dir',
                        required=True,
                        help='Directory to output the prediction results.')
    parser.add_argument('--last_tokens_prediction_ratio',
                        type=float,
                        default=0.5,
                        help='Ratio of how many final tokens to predict (Default = 0.5).')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=128,
                        help='Sequence maximum number of tokens for evaluation (Default = 128).')
    parser.add_argument('--min_seq_len',
                        type=int,
                        default=5,
                        help='Sequence minimum number of tokens for evaluation (Default = 5).')
    parser.add_argument('--save_intermediate_predictions',
                        action='store_true',
                        default=False,
                        help='Save intermediate predictions (default = False).')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information (default = False).')

    return parser.parse_args()


class HighwayNetwork(torch.nn.Module):

    def __init__(self,
                 size,
                 num_layers = 1,
                 non_linearity = torch.relu,
                 transform_gate_bias_offset = -2, # according to papers https://arxiv.org/abs/1508.06615 and https://arxiv.org/abs/1507.06228
                ):
        super(HighwayNetwork, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # parameters
        self.size = size
        self.num_layers = num_layers
        self.non_linearity = non_linearity
        self.transform_gate_bias_offset = transform_gate_bias_offset

        # layers
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({'transform': torch.nn.Linear(self.size, self.size),
                                 'transform_gate': torch.nn.Linear(self.size, self.size),
                                }) for _ in range(self.num_layers)
                                          ])

    def forward(self, x):
        for layer in self.layers:
            output = layer['transform'](x)
            transform_gate = torch.sigmoid(layer['transform_gate'](x) + self.transform_gate_bias_offset)
            x = transform_gate*output + (1-transform_gate)*x
        return x


class CharacterLanguageModel(torch.nn.Module):
    # TODO - Gradient clipping ||.|| = 5
    # TODO - double-check paper model description

    def __init__(self,
                 char_vocabulary_size,
                 char_embedding_dim,
                 char_padding_idx,
                 char_convolutional_filter_numbers,
                 char_convolutional_filter_sizes,
                 highway_layers,
                 lstm_layers,
                 lstm_dim,
                 dropout_prob,
                 word_vocabulary_size,
                ):
        super(CharacterLanguageModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # parameters
        self.char_vocabulary_size = char_vocabulary_size
        self.char_embedding_dim = char_embedding_dim
        self.char_padding_idx = char_padding_idx
        if len(char_convolutional_filter_numbers) != len(char_convolutional_filter_sizes):
            raise Exception(f'Length mismatch between char_convolutional_filter_numbers ({len(char_convolutional_filter_numbers)}) and char_convolutional_filter_sizes ({len(char_convolutional_filter_sizes)}) parameters.')
        self.char_convolutional_filter_numbers = char_convolutional_filter_numbers
        self.char_convolutional_filter_sizes = char_convolutional_filter_sizes
        self.highway_layers = highway_layers
        self.lstm_layers = lstm_layers
        self.lstm_dim = lstm_dim
        self.dropout_prob = dropout_prob
        self.word_vocabulary_size = word_vocabulary_size

        # layers
        self.char_embedding = torch.nn.Embedding(num_embeddings=self.char_vocabulary_size,
                                                 embedding_dim=self.char_embedding_dim,
                                                 padding_idx=self.char_padding_idx,
                                                )

        self.char_convolutional_filters = torch.nn.ModuleList([
            # TODO - check default parameters
            torch.nn.Conv1d(in_channels=self.char_embedding_dim,
                            out_channels=self.char_convolutional_filter_numbers[idx],
                            kernel_size=self.char_convolutional_filter_sizes[idx],
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            bias=True,
                            padding_mode='zeros',
                           ) for idx in range(len(self.char_convolutional_filter_numbers))
                                                              ])

        feature_vector_size = numpy.sum(self.char_convolutional_filter_numbers)

        self.highway_network = HighwayNetwork(size=feature_vector_size,
                                              num_layers=self.highway_layers,
                                             )

        self.lstm = torch.nn.LSTM(input_size=feature_vector_size,
                                  hidden_size=self.lstm_dim,
                                  num_layers=self.lstm_layers,
                                  dropout=self.dropout_prob if self.lstm_layers > 1 else 0,     # there is no dropout in the last layer
                                  batch_first=True,
                                  bidirectional=False,
                                 )
        self.dropout = torch.nn.Dropout(self.dropout_prob)    # there is no dropout in the last LSTM layer
        self.logits = torch.nn.Linear(self.lstm_dim, self.word_vocabulary_size)

    def forward(self, x):

        # reordering the dimensions to help iterate through convolutional filters (token-wise): (batch size, sequence length, token length) to (sequence length, batch size, token length)
        x = x.permute(1, 0, 2)
        
        # char embeddings
        x = [ self.char_embedding(token) for token in x ]
        
        # list of outputs from convolutional filters
        x = [ [ torch.tanh(conv(token.permute(0, 2, 1))) for conv in self.char_convolutional_filters ] for token in x ]
        
        # list of outputs from pooling operations
        x = [ [ torch.nn.functional.max_pool1d(output, kernel_size=output.shape[2]).squeeze(dim=2) for output in token ] for token in x ]

        # concatenation of pooling results into a single vector for each token
        x = [ torch.cat(token, dim=1) for token in x ]
        
        # output of highway network for each token
        x = [ self.highway_network(token) for token in x ]

        # bringing the dimensions to the original ordering (batch-first): (sequence length, batch size, token length) to (batch size, sequence length, token length) 
        x = torch.stack(x).permute(1, 0, 2)
        
        # LSTM layer output
        batch_size = x.shape[0]
        x, _ = self.lstm(x)
        
        x = self.dropout(x[:,-1,:].view(batch_size, -1))    # lstm_out[:,-1,:] => last token output features from the last LSTM layer
        
        x = self.logits(x)
        
        return x


def char_encode_sample(sample, tokenizer, char_vocab, fixed_length=0):
    sample_char_encoded = []
    tokens = tokenizer.convert_ids_to_tokens(sample)
    if not fixed_length:
        fixed_length = max([ len(token) for token in tokens ]) + 2  # room for [sot] and [eot] markers
    for token in tokens:
        char_encoded_token = [char_vocab['[sot]']] + [ char_vocab.get(ch, char_vocab['[unk]']) for ch in token ] + [char_vocab['[eot]']]
        if len(char_encoded_token) > (fixed_length):
            char_encoded_token = char_encoded_token[:fixed_length-1] + [char_vocab['[eot]']]
        else:
            char_encoded_token += [ char_vocab['[pad]'] ] * (fixed_length - len(char_encoded_token))
        sample_char_encoded.append(char_encoded_token)
    return sample_char_encoded


if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()
    
    logger_format = '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=logger_format)
    logger = logging.getLogger(__name__)

    if os.path.exists(args.output_dir):
        logger.error(f'Output directory {args.output_dir} already exists. Quitting ...')
        sys.exit(1)
    os.mkdir(args.output_dir)

    # logging file configuration
    logger_file_handler = logging.FileHandler(os.path.join(args.output_dir, 'main.log'), encoding='utf-8')
    logger_file_handler.setFormatter(logging.Formatter(logger_format))
    logger.addHandler(logger_file_handler)
    
    logger.info('Output directory created at {} ...'.format(args.output_dir))

    if args.debug:
        logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
    else:
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)


    if torch.cuda.is_available():
        logger.info('CUDA available. Using GPU ...')
        device = torch.device('cuda:0')
    else:
        logger.info('CUDA unavailable. Using CPU ...')
        device = torch.device('cpu')

    logger.info('Starting Character LSTM language model evaluation with the following parameters:\n{}'.format(pprint.pformat(vars(args))))
    with open(os.path.join(args.output_dir, 'arguments.json'), mode='xt', encoding='utf-8') as fd:
        json.dump(vars(args), fd, ensure_ascii=True, sort_keys=True)

    logger.info('Loading model training parameters ...')
    with open(os.path.join(args.models_dir, 'arguments.json'), mode='rt', encoding='utf-8') as fd:
        training_args = json.load(fd)
        training_args = collections.namedtuple('Parameters', training_args.keys())(*training_args.values())

    logger.info('Loading tokenizer ...')
    word_tokenizer = transformers.BertTokenizer.from_pretrained(args.models_dir)
    special_tokens_filename = os.path.join(args.models_dir, 'special_tokens.json')
    if os.path.exists(special_tokens_filename):
        with open(special_tokens_filename, mode='rt', encoding='utf-8') as fd:
            special_tokens = json.load(fd)
        number_added_tokens = word_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        logger.debug(f'\t{number_added_tokens} special tokens loaded from {special_tokens_filename}.')

    authors_ids = sorted([ os.path.basename(author_dir) for author_dir in glob.glob(os.path.join(args.tweets_dir, '[0-9]*')) ])
    joblib.dump(authors_ids, os.path.join(args.output_dir, 'authors_ids.joblib'))

    logger.info('Loading and tokenizing the dataset ...')
    authors_messages = []
    for author_id in authors_ids:
        with open(os.path.join(args.tweets_dir, author_id, 'valid.json'), mode='rt', encoding='utf-8') as fd:
            messages = json.load(fd)
        for message in messages:
            message['input_ids'] = word_tokenizer(message['text'],
                                                  max_length=args.max_seq_len,
                                                  truncation=True,
                                                  padding=False,
                                                  return_token_type_ids=False,
                                                  return_attention_mask=False,
                                                 )['input_ids']
            message['idxs'] = list(range(len(message['input_ids'])//2, len(message['input_ids'])))
            message['predictions'] = numpy.zeros((len(message['input_ids']) - len(message['input_ids'])//2 ,
                                                  len(authors_ids),
                                                 ))
        authors_messages.append(messages)
        if args.save_intermediate_predictions:
            joblib.dump(authors_messages, os.path.join(args.output_dir, 'authors_messages_intermediate.joblib'))

    logger.info('Running predictions ...')

    for author_idx, author_id in enumerate(authors_ids):
        logger.debug(f'Predicting tokens using language model for author {author_id} ({author_idx+1}/{len(authors_ids)}) ...')

        author_dir =  os.path.join(args.models_dir, author_id)

        with open(os.path.join(author_dir, 'char_vocabulary.json'), mode='rt', encoding='utf-8') as fd:
            char_vocabulary = json.load(fd)

        model = CharacterLanguageModel(char_vocabulary_size=len(char_vocabulary),
                                       char_embedding_dim=training_args.char_embedding_dim,
                                       char_padding_idx=char_vocabulary['[pad]'],
                                       char_convolutional_filter_numbers=training_args.char_convolutional_filter_numbers,
                                       char_convolutional_filter_sizes=training_args.char_convolutional_filter_sizes,
                                       highway_layers=training_args.highway_layers,
                                       lstm_layers=training_args.lstm_layers,
                                       lstm_dim=training_args.lstm_dim,
                                       dropout_prob=training_args.dropout_prob,
                                       word_vocabulary_size=len(word_tokenizer),
                                      )

        model.load_state_dict(torch.load(os.path.join(author_dir, 'best_model.pt'))['model_state_dict'])
        model.eval()
        model.to(device)

        with torch.no_grad():
            for author_messages_idx, author_messages in enumerate(authors_messages):
                #logger.debug(f'\tPredicting messages from author {authors_ids[author_messages_idx]} ({author_messages_idx+1}/{len(authors_messages)}) ...')
                for message_dict in author_messages:
                    #logger.debug(f'Original message to be predicted:\n{message_dict["text"]}')
                    for idx, token_idx in enumerate(message_dict['idxs']):

                        sample = message_dict['input_ids'][:token_idx]
                        #TODO use original token_max_len
                        sample_char_encoded = char_encode_sample(sample, word_tokenizer, char_vocabulary)
                        model_input = torch.tensor(sample_char_encoded, dtype=torch.long).unsqueeze(0).to(device)
                        lm_probs = torch.nn.functional.softmax(model(model_input), dim=1)[0]
                        lm_probs_argsorted = lm_probs.argsort()
                        message_dict['predictions'][idx, author_idx] = lm_probs[message_dict['input_ids'][token_idx]].item()

                        #logger.debug(f'\t\tTarget index: {token_idx}')
                        #logger.debug(f'\t\tTrue token: {word_tokenizer.convert_ids_to_tokens(message_dict["input_ids"][token_idx])}')
                        #logger.debug(f'\t\tPredicted token: {word_tokenizer.convert_ids_to_tokens(lm_probs_argsorted[-1].item())}')
        
        del model

        if args.save_intermediate_predictions:
            partial_predictions = []
            for author_messages_idx, author_messages in enumerate(authors_messages):
                partial_predictions_inner = []
                for message_dict in author_messages:
                    partial_predictions_inner.append(message_dict['predictions'][:, author_idx])
                partial_predictions.append(partial_predictions_inner)
            joblib.dump(partial_predictions, os.path.join(args.output_dir, f'authors_messages_intermediate_author-{authors_ids[author_idx]}_author-id-{author_idx}.joblib'))

    # move from a distribution of probabilities over the vocabulary to a distribution of probabilities over the authors
    # no difference if using the probabilities ordering to predict the author since the ordering doesn't change
    #logger.info('Transforming the probabilitities distributions ...')
    #for author_messages in authors_messages:
    #    for message_dict in author_messages:
    #        for token_idx in range(message_dict['predictions'].shape[0]):    # for each token
    #            row_sum = message_dict['predictions'][token_idx,:].sum()
    #            message_dict['predictions'][token_idx,:] /= row_sum
                
    for author_messages in authors_messages:
        for message_dict in author_messages:
            acc = numpy.zeros(message_dict['predictions'].shape[1])
            for token_idx in range(message_dict['predictions'].shape[0]):
                sorted_authors_idxs = numpy.argsort(message_dict['predictions'][token_idx])
                for idx, author_idx in enumerate(sorted_authors_idxs):
                    acc[author_idx] += idx
            message_dict['predictions_sorted'] = numpy.argsort(acc)
            
    logger.info(f'Saving data to {args.output_dir} ...')
    joblib.dump(authors_messages, os.path.join(args.output_dir, 'authors_messages.joblib'))

    logger.info('Calculating accuracies ...')
    authors_messages = joblib.load(os.path.join(args.output_dir, 'authors_messages.joblib'))
    accuracy = []
    for i in range(1, 51):
        wins = 0
        losses = 0
        ignored_messages = 0
        for author_idx, author_messages in enumerate(authors_messages):
            for message_dict in author_messages:
                if len(message_dict['input_ids']) < args.min_seq_len + 2:  # accounting for [CLS] and [SEP] tokens
                    ignored_messages += 1
                    continue
                if author_idx in set(message_dict['predictions_sorted'][-i:]):
                    wins += 1
                else:
                    losses += 1
        accuracy.append(wins/(wins+losses))
        logger.info(f'Top {i} results: Wins = {wins}; Losses = {losses}; Accuracy = {accuracy[-1]} (ignored messages = {ignored_messages})')

    logger.info('Finished.')
