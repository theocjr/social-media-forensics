#!/usr/bin/env python3


import argparse
import logging
import transformers
import pprint
import os
import sys
import torch
import json
import datasets
import glob
import shutil


def command_line_parsing():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--model_output_dir',
                        required=True,
                        help='Directory to output the trained model.')
    parser.add_argument('--training_filename',
                        required=True,
                        help='Training filename (one line per sample).')
    parser.add_argument('--valid_filename',
                        required=True,
                        help='Validation filename (one line per sample).')
    parser.add_argument('--test_filename',
                        required=True,
                        help='Test filename (one line per sample).')
    parser.add_argument('--from_pre_trained',
                        help='Identifier to load pre-trained models.')
#    parser.add_argument('--vocabulary_size',
#                        type=int,
#                        default=30_000,
#                        help='Vocabulary size (Default = 30,000).')
    parser.add_argument('--finetune_prefix_layers',
                        nargs='*',
                        help='List of perfix layers to fine-tune. Example: \'bert.encoder.layer.11. cls.predictions.\'.')
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
    parser.add_argument('--delete_checkpoints',
                        action='store_true',
                        default=False,
                        help='Delete the intermediate checkpoints (default = False).')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='Print debug information (default = False).')
    return parser.parse_args()



if __name__ == '__main__':
    # parsing arguments
    args = command_line_parsing()

    # logging configuration
    logger_format = '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=logger_format)
    logger = logging.getLogger(__name__)
    if args.debug:
        transformers.tokenization_utils.logger.setLevel(logging.INFO)
        transformers.configuration_utils.logger.setLevel(logging.INFO)
        transformers.modeling_utils.logger.setLevel(logging.INFO)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
    else:
        transformers.tokenization_utils.logger.setLevel(logging.WARNING)
        transformers.configuration_utils.logger.setLevel(logging.WARNING)
        transformers.modeling_utils.logger.setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

    if os.path.exists(args.model_output_dir):
        logger.error(f'Output directory {args.model_output_dir} already exists. Quitting ...')
        sys.exit(1)
    os.mkdir(args.model_output_dir)

    # logging file configuration
    logger_file_handler = logging.FileHandler(os.path.join(args.model_output_dir, 'training.log'), encoding='utf-8')
    logger_file_handler.setFormatter(logging.Formatter(logger_format))
    logger.addHandler(logger_file_handler)
    transformers.utils.logging.get_logger().addHandler(logger_file_handler)
    logger.info('Destination directory created at {} ...'.format(args.model_output_dir))

    logger.info('Starting BERT model training with the following parameters:\n{}'.format(pprint.pformat(vars(args))))

    if torch.cuda.is_available():
        logger.info('CUDA available. Using GPU ...')
        device = torch.device('cuda:0')
    else:
        logger.info('CUDA unavailable. Using CPU ...')
        device = torch.device('cpu')

    if args.from_pre_trained:
        logger.info(f'Loading the tokenizer {args.from_pre_trained} ...')
        tokenizer = transformers.BertTokenizer.from_pretrained(args.from_pre_trained)
        logger.info(f'Tokenizer loaded. Current size: {len(tokenizer)}')                                                                                      
        logger.info(f'Loading the model {args.from_pre_trained} ...')
        model = transformers.BertForMaskedLM.from_pretrained(args.from_pre_trained)
        model.to(device)
        logger.debug(f'Model loaded. Architecture:\n\n{model}\n\n')

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
            model.resize_token_embeddings(len(tokenizer))
            with open(os.path.join(args.model_output_dir, 'special_tokens.json'), mode='xt', encoding='utf-8') as fd:
                json.dump(special_tokens, fd, ensure_ascii=True)

        tokenizer.save_vocabulary(args.model_output_dir)

    else:
        # NOT TESTED/USED
        if not os.path.exists(os.path.join(args.model_output_dir, 'vocab.txt')):
            logger.info('Training the tokenizer ...')
            # WordPiece Tokenizer
            tokenizer = tokenizers.BertWordPieceTokenizer()
            tokenizer.train(files=args.training_filename,
                            vocab_size=args.vocabulary_size,
                            )
            logger.info('Saving the tokenizer ...')
            tokenizer.save(args.model_output_dir)
        logger.info('Loading the tokenizer ...')
        tokenizer = transformers.BertTokenizerFast.from_pretrained(args.model_output_dir, model_max_len=100)
        logger.info('Building the model ...')
        config = transformers.BertConfig(vocab_size=args.vocabulary_size,
                                            max_position_embeddings=128,
                                            type_vocab_size=1,
                                        )
        model = transformers.BertForMaskedLM(config=config)

    if args.finetune_prefix_layers:
        logger.info('Fine-tune parameters for layers with prefix(es) {} ...'.format(args.finetune_prefix_layers))
        prefix_layers_to_finetune = tuple(args.finetune_prefix_layers)
        for name, param in model.named_parameters():
            if not name.startswith(prefix_layers_to_finetune):
                logger.debug('\tFreezing parameters for layer {} ...'.format(name))
                param.requires_grad = False

    logger.info('Loading the dataset ...')
    dataset = datasets.load_dataset('text', data_files={'training': args.training_filename,
                                                        'valid': args.valid_filename,
                                                        'test': args.test_filename,
                                                       })

    logger.info('Tokenizing the dataset ...')
    dataset = dataset.map(lambda data: tokenizer(data['text'],
                                                 max_length = model.config.max_position_embeddings,
                                                 truncation=True,
                                                 padding=False,
                                                ),
                          batched=True,
                          batch_size=1000,
                         )

    logger.info('Calculating dataset lengths (token-based) ...')
    sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
    dataset = dataset.map(lambda data: { 'length': data['input_ids'].index(sep_token_id) + 1 })

    dataset.set_format(type='torch',
                       columns=['input_ids',
                                'token_type_ids',
                                'attention_mask',
                                'length',
                                ],
                       device=device,
                      )

    logger.info('Training the model ...')
    training_args = transformers.TrainingArguments(output_dir=args.model_output_dir,
                                                   per_device_train_batch_size=args.batch_size,
                                                   per_device_eval_batch_size=args.batch_size,
                                                   num_train_epochs=args.max_epochs,
                                                   evaluation_strategy='epoch',
                                                   logging_strategy='epoch',
                                                   logging_dir=os.path.join(args.model_output_dir, 'tensorboard_log'),
                                                   save_strategy='epoch',
                                                   save_total_limit=args.early_stop+1,
                                                   load_best_model_at_end=True,
                                                   metric_for_best_model='loss',
                                                   greater_is_better=False,
                                                   group_by_length=True,
                                                   length_column_name='length',
                                                  )
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                 mlm=True,
                                                                 mlm_probability=0.15,
                                                                 return_tensors='pt',
                                                                )
    trainer = transformers.Trainer(model=model,
                                   args=training_args,
                                   train_dataset=dataset['training'],
                                   eval_dataset=dataset['valid'],
                                   data_collator=data_collator,
                                   callbacks = [ transformers.EarlyStoppingCallback(early_stopping_patience=args.early_stop) ],
                                  )
    train_result = trainer.train()

    logger.info('Saving model, tokenizer, and results ...')
    with open(os.path.join(args.model_output_dir, 'training_metrics.json'), mode='wt', encoding='utf-8') as fd:
        json.dump(train_result, fd, ensure_ascii=True, sort_keys=True)

    best_model_dir = os.path.join(args.model_output_dir, 'best_model')
    trainer.save_model(best_model_dir)
    shutil.copy2(os.path.join(args.model_output_dir, 'vocab.txt'), best_model_dir)
    shutil.copy2(os.path.join(args.model_output_dir, 'special_tokens.json'), best_model_dir)

    logger.info('Running evaluation on test data ...')

    logger.info(f'Loading the tokenizer from {best_model_dir} ...')
    tokenizer = transformers.BertTokenizer.from_pretrained(best_model_dir)
    logger.info(f'Tokenizer loaded from {best_model_dir}. Current size: {len(tokenizer)}')
    special_tokens_filename = os.path.join(best_model_dir, 'special_tokens.json')
    if os.path.exists(special_tokens_filename):
        with open(special_tokens_filename, mode='rt', encoding='utf-8') as fd:
            special_tokens = json.load(fd)
        number_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if number_added_tokens > 0:
            logger.info(f'{number_added_tokens} special tokens added to the vocabulary. New size: {len(tokenizer)}')

    logger.info(f'Loading the model from {best_model_dir} ...')
    model = transformers.BertForMaskedLM.from_pretrained(best_model_dir)
    model.to(device)
    logger.debug(f'Model loaded from {best_model_dir} . Architecture:\n\n{model}\n\n')

    logger.info('Predicting over test data ...')
    test_dir = os.path.join(args.model_output_dir, 'test')
    test_args = transformers.TrainingArguments(output_dir=test_dir,
                                               eval_accumulation_steps=args.batch_size,
                                              )
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                 mlm=True,
                                                                 mlm_probability=0.15,
                                                                 return_tensors='pt',
                                                                )
    trainer = transformers.Trainer(model=model,
                                   args=test_args,
                                   data_collator=data_collator,
                                  )
    test_result = trainer.predict(dataset['test'])
    logger.info('\tTest loss: {}'.format(test_result.metrics['test_loss']))
    with open(os.path.join(args.model_output_dir, 'test_metrics.json'), mode='xt', encoding='utf-8') as fd:
        json.dump(train_result, fd, ensure_ascii=True, sort_keys=True)

    logger.info(f'Loading fill-mask pipeline using model at {best_model_dir} ...')
    fill_mask = transformers.pipeline('fill-mask',
                                      model=best_model_dir,
                                      tokenizer=tokenizer, # if loaded from the directory, need to add the additional special tokens
                                     )
    test_sentence = 'First [MASK] in Washington.'
    logger.info('Result for the sentence: \"{}\":\n{}'.format(test_sentence, pprint.pformat(fill_mask(test_sentence))))


    logging.info('Cleaning up ...')
    shutil.rmtree(test_dir) # empty directory not needed
    if args.delete_checkpoints:
        checkpoint_dirs = glob.glob(os.path.join(args.model_output_dir, 'checkpoint-*'))
        for checkpoint_dir in checkpoint_dirs:
            shutil.rmtree(checkpoint_dir)

    logger.info('Finished.')
