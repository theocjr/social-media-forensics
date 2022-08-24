#!/usr/bin/env python3


import sys
import itertools


if len(sys.argv) != 5:
    print('ERROR - Usage {} <source-dir> <number-classes> <vocabulary-size> <logfile-prefix>')
    sys.exit(1)

params_values = {
    'learning_rate': [0.1],
    #'word_vector_size': [24, 48, 72],
    'filter_size_conv_layers': list(itertools.product([5, 10, 30], [4, 7, 21])),
    'nr_of_filters_conv_layers':list(itertools.product([3, 6, 18], [6, 12, 36])),
    'ktop': [2, 4, 12],
}

cmd_options = []
for param in sorted(params_values.keys()):
    cmd_option = []
    for value in params_values[param]:
        if isinstance(value, tuple):
            cmd_option.append(''.join([ '--', param, ' ', ' '.join( [ str(item) for item in value ] ) ]))
        else:
            cmd_option.append(''.join([ '--', param, ' ', str(value) ]))
    cmd_options.append(cmd_option)

cmd_init = ' '.join(['./classify.py',
                     '--debug',
                     '--max_minutes 2880',
                     '--source-dir-data',  sys.argv[1],
                     '--output_classes', sys.argv[2],
                     '--vocab_size', sys.argv[3],
                    ])
cmd_end = '> /dev/null 2>'
for run_options in itertools.product(*cmd_options):
    log_id = ['./logs/' + sys.argv[4]]
    for option in run_options:
        log_id.append( option.replace('--', '').replace(' ', '-').replace('_','-') )
    log_id.append('.log')
    log_filename = '_'.join(log_id)
    print(' '.join( [cmd_init] + list(run_options) + [cmd_end, log_filename, '&'] ))
    print('sleep 120')
