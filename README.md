# Social Media Forensics

This project aims to make available tools and software for forensics over social media.

## Datasets

TODO

## Installation

TODO: Describe the installation process

## Usage

### Microblog Authorship Attribution

Code related to tackle the problem of authorship attribution over micro blog social media like Twitter.

The code is organized into these three directories:

* `dataset_pre_processing`: code that pre-processes the data before to trigger the model learning. Examples of pre-processing are language filtering, data tagging and n-grams generation.
* `classification`: code to learn a model and run a classifier.
* `experiment_scripts`: examples of scripts to run the code in the above directories.
* `char-grams_analysis`: code and scripts related to analysis of different char-grams.

Each Python program has a `-h/--help` command-line option that presents an usage explanation describing each option meaning.

#### Dataset Pre-Processing

The `dataset_pre_processing` directory contains code to act over the dataset in order to prepare it to be run by code located in `classification` directory. All the code are optional except `ngrams_generator.py` that must be run before triggering the classifiers. A normal order of invocation is:

1. `filter_retweets_few_words.py`
2. `filter_language_by_tweet.py`
3. `tagging_irrelevant_data.py`
4. `ngrams_generator.py`

After running `ngrams_generator.py` the output is ready to be run by the classifiers presented in the `classification` directory.


## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

TODO: Write credits

## License

TODO: Write license
