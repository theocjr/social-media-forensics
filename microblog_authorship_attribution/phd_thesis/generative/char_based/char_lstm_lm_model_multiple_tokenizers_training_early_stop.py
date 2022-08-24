#!/usr/bin/env python
# coding: utf-8


import os
import torch
import argparse
import logging
import transformers
import sys
import json
import pprint
import numpy
import datasets

sys.path.append(os.path.join('..', '..', '..', 'utils'))
import preprocessing_json

import string
import tokenizers
import random
import shutil
import glob

import torch
import argparse
import logging
import os
import sys
import json
import pprint
import transformers
import glob
import datasets
import random
import shutil


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)

    # data
    parser.add_argument('--input_dir',
                        required=True,
                        help='Input directory (one sub-directory per author).')
    parser.add_argument('--output_dir',
                        required=True,
                        help='Directory to output the trained models.')

    # architecture
    parser.add_argument('--char_embedding_dim',
                        type=int,
                        default=15,
                        help='Dimension of the character embeddings layer (Default = 15).')
    parser.add_argument('--char_convolutional_filter_numbers',
                        nargs='+',
                        type=int,
                        default=[100, 150, 150, 200, 200, 200],
                        help='List of numbers of character convolutional filters (length must match --char_convolutional_filter_sizes option). Default=[100, 150, 150, 200, 200, 200].')
    parser.add_argument('--char_convolutional_filter_sizes',
                        nargs='+',
                        type=int,
                        default=[1, 2, 3, 4, 5, 6],
                        help='List of sizes of character convolutional filters (length must match --char_convolutional_filter_numbers option). Default=[1, 2, 3, 4, 5, 6].')
    parser.add_argument('--lstm_layers',
                        type=int,
                        default=2,
                        help='Number of LSTM layers (stacked) (Default = 2).')
    parser.add_argument('--lstm_dim',
                        type=int,
                        default=650,
                        help='Dimension of the hidden LSTM layer (Default = 650).')
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=0.5,
                        help='Dropout probability (Default = 0.5).')
#    parser.add_argument('--highway_layers',
#                        type=int,
#                        default=2,
#                        help='Number of Highway Network layers (stacked) (Default = 2).')

    # training parameters
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        help='Learning rate (Default = 0.1).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='Batch size (Default = 20).')
    parser.add_argument('--max_epochs',
                        type=int,
                        default=1,
                        help='Maximum number of epochs to train (Default = 1).')
    parser.add_argument('--early_stop',
                        type=int,
                        default=0,
                        help='Number of epochs without improvement to stop the training early (Default = 0).')
    parser.add_argument('--min_seq_len',
                        type=int,
                        default=5,
                        help='Sequence minimum number of tokens for training (Default = 5).')

    # pre-processing
    parser.add_argument('--tag_url',
                        action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--tag_user',
                        action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--tag_number',
                        action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--tag_hashtag',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--demojize',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--textify_emoji',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--mention_limit',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--punc_limit',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--lower_hashtag',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--segment_hashtag',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--lower_case',
                        action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--add_capital_signs',
                        action='store_true',
                        default=False,
                        help='')

    # misc
    parser.add_argument('--delete_checkpoints',
                        action='store_true',
                        default=False,
                        help='Delete the intermediate checkpoints (default = False).')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information (default = False).')

    return parser.parse_args()


class LSTMCharLM(torch.nn.Module):

    # TODO - Highway network
    # TODO - Gradient clipping ||.|| = 5
    # TODO - Parameter initialization [-0.05, 0.05]
    # TODO - double-check paper model description

    def __init__(self,
                 char_vocabulary_size,
                 char_embedding_dim,
                 char_padding_idx,
                 char_convolutional_filter_numbers,
                 char_convolutional_filter_sizes,
                 lstm_layers,
                 lstm_dim,
                 dropout_prob,
                 word_vocabulary_size,
                ):
        super(LSTMCharLM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        # parameters
        self.char_vocabulary_size = char_vocabulary_size
        self.char_embedding_dim = char_embedding_dim
        self.char_padding_idx = char_padding_idx
        if len(char_convolutional_filter_numbers) != len(char_convolutional_filter_sizes):
            raise Exception(f'Length mismatch between char_convolutional_filter_numbers ({len(char_convolutional_filter_numbers)}) and char_convolutional_filter_sizes ({len(char_convolutional_filter_sizes)}) parameters.')
        self.char_convolutional_filter_numbers = char_convolutional_filter_numbers
        self.char_convolutional_filter_sizes = char_convolutional_filter_sizes
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

        self.lstm = torch.nn.LSTM(input_size=numpy.sum(self.char_convolutional_filter_numbers),
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
    for token in tokenizer.convert_ids_to_tokens(sample):
        char_encoded_token = [char_vocab['[sot]']] + [ char_vocab[ch] for ch in token ] + [char_vocab['[eot]']]
        if fixed_length:
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

    with open(os.path.join(args.output_dir, 'arguments.json'), mode='xt', encoding='utf-8') as fd:
        json.dump(vars(args), fd, ensure_ascii=True, sort_keys=True)
    logger.info('Starting Character LSTM language model training with the following parameters:\n{}'.format(pprint.pformat(vars(args))))

    author_ids = sorted([ os.path.basename(author_dir) for author_dir in glob.glob(os.path.join(args.input_dir, '[0-9]*')) ])
    for author_idx, author_id in enumerate(author_ids):

        logger.info(f'Processing author {author_id} {author_idx+1}/{len(author_ids)} ...')

        author_dir = os.path.join(args.output_dir, author_id)
        os.mkdir(author_dir)

        author_logger = logging.getLogger(f'lm_{author_id}')
        author_logger_file_handler = logging.FileHandler(os.path.join(author_dir, 'training.log'), encoding='utf-8')
        author_logger_file_handler.setFormatter(logging.Formatter(logger_format))
        author_logger.addHandler(author_logger_file_handler)

        author_logger.info('Loading the dataset ...')
        dataset = datasets.load_dataset('text', data_files={'training': os.path.join(args.input_dir, author_id, 'training_text.txt'),
                                                            'valid': os.path.join(args.input_dir, author_id, 'valid_text.txt'),
                                                            #'test': os.path.join(args.input_dir, author_id, 'test_text.txt'),
                                                        })

        author_logger.info('Pre-processing text ...')
        preprocessor = preprocessing_json.build_preprocess(tag_url=args.tag_url,
                                                        tag_user=args.tag_user,
                                                        tag_number=args.tag_number,
                                                        tag_hashtag=args.tag_hashtag,
                                                        demojize=args.demojize,
                                                        textify_emoji=args.textify_emoji,
                                                        mention_limit=args.mention_limit,
                                                        punc_limit=args.punc_limit,
                                                        lower_hashtag=args.lower_hashtag,
                                                        segment_hashtag=args.segment_hashtag,
                                                        lower_case=args.lower_case,
                                                        add_capital_signs=args.add_capital_signs,
                                                        )
        dataset = dataset.map(lambda data: {'preprocessed': preprocessor(data['text'])})

        author_logger.info('Building word vocabulary ...')
        special_tokens = ['<http>',
                        '<user>',
                        '<number>',
                        '<emoji>', '</emoji>',
                        '<hashtag>', '</hashtag>',
                        '<has_cap>', '<all_cap>',
                        ]
        with open(os.path.join(author_dir, 'special_tokens.json'), mode='xt', encoding='utf-8') as fd:
            json.dump(special_tokens, fd)
        word_tokenizer = tokenizers.BertWordPieceTokenizer(clean_text=True,            # cleans text by removing control characters and replacing all whitespace with spaces
                                                           handle_chinese_chars=True,  # whether the tokenizer includes spaces around Chinese characters
                                                           strip_accents=False,        # whether we remove accents, when True this will make é -> e, ô -> o, etc ...
                                                           lowercase=False,            # if True the tokenizer will view capital and lowercase characters as equal
                                                           wordpieces_prefix='##',     # the prefix added to pieces of words
                                                          )
        word_tokenizer.train_from_iterator(iterator=dataset['training']['preprocessed'],
                                           vocab_size=30_000,                          # maximum number of tokens in the final tokenizer
                                           min_frequency=2,                            # minimum frequency for a pair of tokens to be merged
                                           #limit_alphabet=len(char_vocab),             # maximum number of different characters.
                                           limit_alphabet=1_000,                       # maximum number of different characters.
                                           special_tokens=special_tokens + ['[PAD]',
                                                                            '[UNK]',
                                                                            '[CLS]',
                                                                            '[SEP]',
                                                                            '[MASK]',
                                                                           ],         # a list of the special tokens that BERT uses
                                          )
        author_logger.info(f'Word tokenizer built with {word_tokenizer.get_vocab_size()} tokens.')
        with open(os.path.join(author_dir, 'config_tokenizer.json'), mode='xt', encoding='utf-8') as fd:
            json.dump({'model_type': 'bert'}, fd)
        word_tokenizer.save_model(author_dir)
        author_logger.info(f'Word tokenizer saved at {author_dir}')

        word_tokenizer = transformers.BertTokenizer.from_pretrained(author_dir,
                                                                    do_lower_case=False,
                                                                    tokenize_chinese_chars=True,
                                                                    strip_accents=False,
                                                                )
        author_logger.info(f'Tokenizer loaded. Size: {len(word_tokenizer)}')
        number_added_tokens = word_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if number_added_tokens > 0:
            author_logger.info(f'{number_added_tokens} special tokens added to the vocabulary. New size: {len(word_tokenizer)}')

        author_logger.info('Tokenizing the dataset ...')
        dataset = dataset.map(lambda data: { 'tokenized': word_tokenizer.tokenize(data['preprocessed']) })

        author_logger.info('Encoding dataset ...')
        dataset = dataset.map(lambda data: word_tokenizer(data['preprocessed'],
                                                          max_length = 128,
                                                          truncation=True,
                                                          padding=False,
                                                          return_token_type_ids=False,
                                                          return_attention_mask=False,
                                                         ),
                              batched=True,
                              batch_size=1000,
                             )

        author_logger.info('Building character vocabulary ...')
        char_vocab = {}
        token_max_len = max( [ len(token) for token in word_tokenizer.additional_special_tokens ] + [5] )
        char_vocab.update({ k:v for v, k in enumerate(string.printable, start=len(char_vocab))})
        char_vocab['[sot]'] = len(char_vocab)    # start-of-token marker
        char_vocab['[eot]'] = len(char_vocab)    # end-of-token marker
        char_vocab['[pad]'] = len(char_vocab)    # pad marker
        for message in dataset['training']:
            for token in message['tokenized']:
                if len(token) > token_max_len:
                    token_max_len = len(token)
                for character in token:
                    if character not in char_vocab:
                        char_vocab[character] = len(char_vocab)
        author_logger.info(f'Character vocabulary size: {len(char_vocab)}')
        author_logger.info(f'Max token length: {token_max_len}')
        char_vocabulary_filename = os.path.join(author_dir, 'char_vocabulary.json')
        with open(char_vocabulary_filename, mode='xt', encoding='utf-8') as fd:
            json.dump(char_vocab, fd, sort_keys=True)
        author_logger.info(f'Character vocabulary saved at {char_vocabulary_filename}')

        author_logger.info('Assembling batches by length ...')
        min_seq_len = args.min_seq_len + 2  # includes [CLS] and [SEP] BERT tokens
        training_batches = []
        valid_batches = []
        #test_batches = []
        for data_id, batches in [('training', training_batches),
                                 ('valid', valid_batches),
                                 #('test', test_batches),
                                ]:
            samples_by_len = {}
            num_ignored_samples = 0
            for sample in dataset[data_id]:
                if len(sample['input_ids']) < min_seq_len:
                    num_ignored_samples += 1
                    continue
                for idx in range(min_seq_len-1, len(sample['input_ids'])):
                    if idx not in samples_by_len:
                        samples_by_len[idx] = []
                    samples_by_len[idx].append([torch.tensor(sample['input_ids'][:idx], dtype=torch.long),
                                                torch.tensor(sample['input_ids'][idx], dtype=torch.long),
                                            ])
            if num_ignored_samples:
                author_logger.warning(f'Ignored {num_ignored_samples} {data_id} samples due to their size (< {min_seq_len}).')
            current_batch = {'x': [], 'y': []}
            for idx in sorted(samples_by_len):
                for sample in samples_by_len[idx]:
                    if len(current_batch['x']) == args.batch_size:
                        current_batch['x'] = torch.nn.utils.rnn.pad_sequence(current_batch['x'],
                                                                            batch_first=True,
                                                                            padding_value=word_tokenizer.pad_token_id,
                                                                            )
                        current_batch['y'] = torch.tensor(current_batch['y'], dtype=torch.long)
                        batches.append(current_batch)
                        current_batch = {'x': [], 'y': []}
                        continue
                    current_batch['x'].append(sample[0])
                    current_batch['y'].append(sample[1])

        author_logger.info('Transforming token input ids into character indices ...')
        for batches in [training_batches,
                        valid_batches,
        #                test_batches,
                    ]:
            for batch in batches:
                char_encoded_batch = []
                for sample in batch['x']:
                    char_encoded_batch.append(char_encode_sample(sample,
                                                                 word_tokenizer,
                                                                 char_vocab,
                                                                 token_max_len,
                                                                ))
                batch['x'] = torch.tensor(char_encoded_batch, dtype=torch.long)

        author_logger.info('Building the model ...')
        model = LSTMCharLM(char_vocabulary_size = len(char_vocab),
                           char_embedding_dim = args.char_embedding_dim,
                           char_padding_idx = char_vocab['[pad]'],
                           char_convolutional_filter_numbers = args.char_convolutional_filter_numbers,
                           char_convolutional_filter_sizes = args.char_convolutional_filter_sizes,
                           lstm_layers = args.lstm_layers,
                           lstm_dim = args.lstm_dim,
                           dropout_prob = args.dropout_prob,
                           word_vocabulary_size = len(word_tokenizer),
                          )
        model.to(device)


        author_logger.info('Starting training ...')

        loss_function = torch.nn.NLLLoss()
        log_softmax_function = torch.nn.functional.log_softmax
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        training_loss_list = []
        valid_loss_list = []
        best_valid_loss = float('Inf')
        best_epoch = None
        no_improvement = 0  # early-stop watchdog
        for epoch in range(1, args.max_epochs+1):
            model.train()
            training_loss = 0
            random.shuffle(training_batches)
            for batch in training_batches:
                model.zero_grad()
                output = model(batch['x'].to(device))
                loss = loss_function(log_softmax_function(output, dim=1),
                                    batch['y'].to(device),
                                    )
                training_loss += loss
                loss.backward()
                optimizer.step()
            training_loss /= len(training_batches)
            training_loss_list.append(training_loss.item())

            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch in valid_batches:
                    output = model(batch['x'].to(device))
                    valid_loss += loss_function(log_softmax_function(output, dim=1),
                                                batch['y'].to(device),
                                            )
            valid_loss /= len(valid_batches)
            valid_loss_list.append(valid_loss.item())

            author_logger.debug(f'Epoch: {epoch}/{args.max_epochs} - Losses: Training = {training_loss} ; Validation = {valid_loss}')

            if valid_loss < best_valid_loss:
                logger.debug(f'\tBetter model found ({valid_loss} < {best_valid_loss}).')
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(author_dir, f'model_epoch-{best_epoch}.pt'))
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == args.early_stop:
                    logger.info(f'Stopping training at epoch {epoch} due to early-stop configuration ({args.early_stop}) ...')
                    break

        best_model_filename = os.path.join(author_dir, 'best_model.pt')
        author_logger.info(f'Saving best model (epoch {best_epoch}) at {best_model_filename} ...')
        shutil.copy2(os.path.join(author_dir, f'model_epoch-{best_epoch}.pt'),
                    best_model_filename)
        torch.save({'training_losses': training_loss_list,
                    'valid_losses': valid_loss_list,
                },
                os.path.join(author_dir, 'final_metrics.pt'),
                )
        if args.delete_checkpoints:
            author_logger.info('Deleting checkpoints ...')
            checkpoint_models = glob.glob(os.path.join(author_dir, 'model_epoch-*'))
            for checkpoint_model in checkpoint_models:
                os.remove(checkpoint_model)

        author_logger.info('Finished.')
        del model

logger.info('Finished.')
