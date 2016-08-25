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

The `dataset_pre_processing` directory contains code to act over the dataset in order to prepare it to be run by code located in `classification` directory. All the code are optional except `ngrams_generator.py` that must be run before triggering the classifiers. The idea is to invoke these scripts in a pipeline version, with flexibility to choose which scripts to run with the desired options. In this pipeline, the output data of a step is the input data of the next (no operation is done over the input data in any code). Different invocation orders can be chosen but a common one is:

1. `filter_retweets_few_words.py`
2. `filter_language_by_tweet.py`
3. `tagging_irrelevant_data.py`
4. `ngrams_generator.py`

After running `ngrams_generator.py` the output is ready to be run by the classifiers presented in the `classification` directory.

Below is a list of each file/directory with a brief explanation and example of invocation (when applicable):

* `messages_persistence.py`: Auxiliary code for manipulating (read and write) Twitter messages in dataset files.
* `guess-language-0.2`: Language detection API available in https://pypi.python.org/pypi/guess-language included in the project for convenience.
* `filter_retweets_few_words.py`: Code for reading authors' tweets filenames and remove retweets and tweets with few words.
  * Example of use: `filter_retweets_few_words.py  --source-dir-data my_input_dir --dest-dir my_output_dir --minimal-number-words 4 --debug`
* `filter_language_by_tweet.py`: Code for reading authors' tweets filenames and filter the messages based on an API of language detection.
  * Example of use: `./filter_language_by_tweet.py --source-dir my_input_dir --dest-dir my_output_dir --language-detection-module ./guess-language-0.2/guess_language/ --debug`
* `tagging_irrelevant_data.py`: Code for tagging irrelevant data as numbers, dates, times, URLs, hashtags and user references. There are command-line options to individually suppress the tagging of each data type.
  * Example of use: `./tagging_irrelevant_data.py --source-dir my_input_dir --dest-dir my_output_dir --debug`
* `ngrams_generator.py`: Code for generating ngrams. The input are the messages presented in the dataset and the output are files in Python’s built-in persistence model format implemented by `sklearn`, one file for each feature of each author (char-4-gram, word-1-gram, word-2-gram, ...) to be fed to the classifiers. This is the only code needed to be run before any classifier in the `classification` directory.
  * Example of use: `./ngrams_generator.py --source-dir my_input_dir --dest-dir my_output_dir --features all --debug`


#### Classification

The `classification` directory contains code related to the classifiers used to learn a model for the authors in the dataset. They expect that the dataset was previously pre-processed by the code in the `dataset_pre_processing` directory (at least `ngrams_generator.py` code) so that their input be the messages n-grams. Below are a list of the files/directories with brief explanations and examples of usage when applicable:

* `PmSVM`: Power Mean SVM classifier written by Jianxin Wu, originally available in https://sites.google.com/site/wujx2001/home/power-mean-svm. The original code was modified to deal with large data.
* `fest`: FEST software written by Nikos Karampatziakis that provides Random Forest classifiers. Available in http://lowrank.net/nikos/fest/ and included here for convenience.
* `feature_vectors_generator.py`: Code to read n-grams data and to output feature vectors in libsvm format.
  * Example of use: `./feature_vectors_generator.py --source-dir-data my_input_dir --output-dir my_output_dir --minimal-number-tweets 1000 --validation-folding 10 --repetitions 10 --number-authors 50 --number-tweets 50 --features all --debug` 
* `pmsvm_classifier.py`:
* `pmsvm_classifier_no_cross.py`:
* `pmsvm_pca_classifier.py`:
* `rf_classifier.py`:
* `rf_classifier_fest.py`: Program to trigger a Random Forest classifier based on FEST code. Its input are the feature vectors output by `feature_vectors_generator.py` code. FEST code needs to be compiled as indicated in http://lowrank.net/nikos/fest/ before `rf_classifier_fest.py` invocation.
  * Example of use: `./rf_classifier_fest.py --source-dir-data my_input_dir --output-dir my_output_dir --number-trees 500 -debug`


#### Experiment Scripts


#### Utils


#### Misc


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
