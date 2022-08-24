#!/usr/bin/env python
# coding: utf-8

# # Pipeline
# 
# 1. **Pre-processing (done previously):** Pre-process author data with the following configuration:
#     1. Keep hashtags
#     1. Split hashtag terms
#     1. Textify emojis
#     1. Tag numbers, urls, and mentions
#     1. Limit consecutive mentions to 3 
#     1. Lower case text
# 1. **Language Model (LM) Training (done previously):** Fine-tune one language model per author
# 1. **Prediction Strategy: (done by this code)**
#     1. Mask all the tokens (one at a time)
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
The elements of the outer list are the set of messages (HuggingFace dataset) of an author (in the same order of `author_ids` list).
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
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information (default = False).')

    return parser.parse_args()


class LSTMLM(torch.nn.Module):

    def __init__(self,
                 vocabulary_size,
                 embedding_dim,
                 lstm_layers,
                 lstm_dim,
                 dropout_prob,
                 padding_idx,
                ):
        super(LSTMLM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # parameters
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.lstm_layers = lstm_layers
        self.lstm_dim = lstm_dim
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx

        # layers
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocabulary_size,
                                            embedding_dim=self.embedding_dim,
                                            padding_idx=self.padding_idx,
                                           )
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim,
                                  hidden_size=self.lstm_dim,
                                  num_layers=self.lstm_layers,
                                  dropout=self.dropout_prob if self.lstm_layers > 1 else 0,     # there is no dropout in the last layer
                                  batch_first=True,
                                  bidirectional=False,
                                 )
        self.dropout = torch.nn.Dropout(self.dropout_prob)    # there is no dropout in the last LSTM layer
        self.logits = torch.nn.Linear(self.lstm_dim, self.vocabulary_size)

    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_out, lstm_hidden = self.lstm(embeddings)
        batch_size = x.shape[0]
        dropout_out = self.dropout(lstm_out[:,-1,:].view(batch_size, -1))    # lstm_out[:,-1,:] => last token output features from the last LSTM layer
        logits = self.logits(dropout_out)
        return logits


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

    logger.info('Starting LSTM language model evaluation with the following parameters:\n{}'.format(pprint.pformat(vars(args))))
    with open(os.path.join(args.output_dir, 'arguments.json'), mode='xt', encoding='utf-8') as fd:
        json.dump(vars(args), fd, ensure_ascii=True, sort_keys=True)

    logger.info('Loading model training parameters ...')
    with open(os.path.join(args.models_dir, 'arguments.json'), mode='rt', encoding='utf-8') as fd:
        training_args = json.load(fd)
        training_args = collections.namedtuple('Parameters', training_args.keys())(*training_args.values())

    logger.info('Loading tokenizer ...')
    tokenizer = transformers.BertTokenizer.from_pretrained(args.models_dir)
    special_tokens_filename = os.path.join(args.models_dir, 'special_tokens.json')
    if os.path.exists(special_tokens_filename):
        with open(special_tokens_filename, mode='rt', encoding='utf-8') as fd:
            special_tokens = json.load(fd)
        number_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        logger.debug(f'\t{number_added_tokens} special tokens loaded from {special_tokens_filename}.')

    authors_ids = sorted([ os.path.basename(author_dir) for author_dir in glob.glob(os.path.join(args.tweets_dir, '[0-9]*')) ])

    logger.info('Loading and tokenizing the dataset ...')
    authors_messages = []
    for author_id in authors_ids:
        with open(os.path.join(args.tweets_dir, author_id, 'valid.json'), mode='rt', encoding='utf-8') as fd:
            messages = json.load(fd)
        for message in messages:
            message['input_ids'] = tokenizer(message['text'],
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

    logger.info('Running predictions ...')

    for author_idx, author_id in enumerate(authors_ids):
        logger.debug(f'Predicting tokens using language model for author {author_id} ({author_idx+1}/{len(authors_ids)}) ...')

        model = LSTMLM(vocabulary_size=len(tokenizer),
                       embedding_dim=training_args.embedding_dim,
                       lstm_layers=training_args.lstm_layers,
                       lstm_dim=training_args.lstm_dim,
                       dropout_prob=training_args.dropout_prob,
                       padding_idx=tokenizer.pad_token_id,
                      )
        model.load_state_dict(torch.load(os.path.join(args.models_dir, author_id, 'best_model.pt')))
        model.eval()
        model.to(device)

        with torch.no_grad():
            for author_messages_idx, author_messages in enumerate(authors_messages):
                #logger.debug(f'\tPredicting messages from author {authors_ids[author_messages_idx]} ({author_messages_idx+1}/{len(authors_messages)}) ...')
                for message_dict in author_messages:
                    #logger.debug(f'Original message to be predicted:\n{message_dict["text"]}')
                    for idx, token_idx in enumerate(message_dict['idxs']):
                        model_input = torch.tensor(message_dict['input_ids'][:token_idx], dtype=torch.long).to(device).view(1,-1)
                        lm_probs = torch.nn.functional.softmax(model(model_input), dim=1)[0]
                        lm_probs_argsorted = lm_probs.argsort()
                        message_dict['predictions'][idx, author_idx] = lm_probs[message_dict['input_ids'][token_idx]].item()

                        #logger.debug(f'\t\tTarget index: {token_idx}')
                        #logger.debug(f'\t\tTrue token: {tokenizer.convert_ids_to_tokens(message_dict["input_ids"][token_idx])}')
                        #logger.debug(f'\t\tPredicted token: {tokenizer.convert_ids_to_tokens(lm_probs_argsorted[-1].item())}')
        
        del model

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
    joblib.dump(authors_ids, os.path.join(args.output_dir, 'authors_ids.joblib'))
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
