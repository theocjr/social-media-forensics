#!/usr/bin/env python
# coding: utf-8


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
    parser.add_argument('--tokenizer_model',
                        default='bert-base-uncased',
                        help='Identifier to load (HuggingFace) pre-trained tokenizer (Default = bert-base-uncased).')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=64,
                        help='Dimension of the embeddings layer (Default = 64).')
    parser.add_argument('--lstm_layers',
                        type=int,
                        default=1,
                        help='Number of LSTM layers (stacked) (Default = 1).')
    parser.add_argument('--lstm_dim',
                        type=int,
                        default=128,
                        help='Dimension of the hidden LSTM layer (Default = 128).')
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=0.0,
                        help='Dropout probability (Default = 0.0 - no dropout).')

    # training parameters
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        help='Learning rate (Default = 0.1).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='Batch size (Default = 8).')
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

    with open(os.path.join(args.output_dir, 'arguments.json'), mode='xt', encoding='utf-8') as fd:
        json.dump(vars(args), fd, ensure_ascii=True, sort_keys=True)
    logger.info('Starting LSTM language model training with the following parameters:\n{}'.format(pprint.pformat(vars(args))))


    logger.info(f'Loading the tokenizer {args.tokenizer_model} ...')
    tokenizer = transformers.BertTokenizer.from_pretrained(args.tokenizer_model)
    logger.info(f'Tokenizer loaded. Current size: {len(tokenizer)}')
    special_tokens = ['<http>',
                      '<user>',
                      '<number>',
                      '<emoji>', '</emoji>',
                      '<hashtag>', '</hashtag>',
                      '<has_cap>', '<all_cap>',
                     ]
    number_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    if number_added_tokens > 0:
        logger.info(f'{number_added_tokens} special tokens added to the vocabulary. New size: {len(tokenizer)}')
        with open(os.path.join(args.output_dir, 'special_tokens.json'), mode='xt', encoding='utf-8') as fd:
            json.dump(special_tokens, fd, ensure_ascii=True)
    tokenizer.save_vocabulary(args.output_dir)

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
        author_logger.info('Tokenizing the dataset ...')
        dataset = dataset.map(lambda data: tokenizer(data['text'],
                                                     max_length = 128,
                                                     truncation=True,
                                                     padding=False,
                                                     return_token_type_ids=False,
                                                     return_attention_mask=False,
                                                    ),
                              batched=True,
                              batch_size=1000,
                             )
        author_logger.info('Calculating dataset lengths (token-based) ...')
        dataset = dataset.map(lambda data: { 'length': data['input_ids'].index(tokenizer.sep_token_id) + 1})

        author_logger.info('Assembling batches by length ...')
        min_seq_len = args.min_seq_len + 2  # includes [CLS] and [SEP] BERT tokens
        training_batches = []
        valid_batches = []
        #test_batches = []
        for data_id, batches in [('training', training_batches),
                                 ('valid', valid_batches),
                                 #('test', valid_batches),
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
                                                                             padding_value=tokenizer.pad_token_id,
                                                                            )
                        current_batch['y'] = torch.tensor(current_batch['y'], dtype=torch.long)
                        batches.append(current_batch)
                        current_batch = {'x': [], 'y': []}
                        continue
                    current_batch['x'].append(sample[0])
                    current_batch['y'].append(sample[1])

        author_logger.info('Building the model ...')
        model = LSTMLM(vocabulary_size=len(tokenizer),
                       embedding_dim=args.embedding_dim,
                       lstm_layers=args.lstm_layers,
                       lstm_dim=args.lstm_dim,
                       dropout_prob=args.dropout_prob,
                       padding_idx=tokenizer.pad_token_id,
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
                author_logger.debug(f'\tBetter model found ({valid_loss} < {best_valid_loss}).')
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(author_dir, f'model_epoch-{best_epoch}.pt'))
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == args.early_stop:
                    author_logger.info(f'Stopping training at epoch {epoch} due to early-stop configuration ({args.early_stop}) ...')
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
