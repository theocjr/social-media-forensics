# Social Media Forensics

This project aims to make available tools and software for forensics over social media in order to promote information exchange and foster the area.


## Datasets

The datasets used are available under request. If you are interested, please send a message to Prof. Dr. Anderson Rocha (anderson.rocha@ic.unicamp.br).

If you want to use your own dataset of messages, the code expects the following premises:

1. The messages of an author must be on a separated file (UTF-8 encoded) named by its Twitter's id number followed by the extension `.dat`. Example: `1111111111.dat`;
2. Each message has the following format:

```
[Twitter user name] [timestamp] [Tweet id] {
[Tweet message (multi-line allowed)]
#POS [POS Tag data] #POS
}
```

For example:

```
author_nickname 2016-08-28 20:21:22 091294878667731268 {
tell me

https://t.co/cerkmoecem
#POS V O

U #POS
}
author_nickname 2016-03-28 20:52:53 091294878667731987 {
@Friend Im gonna bring beans
#POS I L V V#POS
}
```

More clarification can be obtained inspecting [`messages_persistence.py`](./microblog_authorship_attribution/dataset_pre_processing/messages_persistence.py) file. If you don't plan to use the POS Tag features you can let the `[POS Tag data]` field empty.


## Installation

Besides the Python interpreter (tested under version 2.7.12), to run this code some required libraries have to be installed:

* numpy
* scipy
* nltk
* scikit-learn

A script called [`install_libraries.sh`](./microblog_authorship_attribution/utils/install_libraries.sh) in `utils` directory is able to automatically perform this install in a fresh new Ubuntu Server LTS 64 14.04 (Trusty Tahr).

Besides these, the FEST code included in this bundle needs to be compiled if the Random Forest classifier based on it is supposed to be run (see details in [Classification Subsection](#classification)).


## Usage


### Microblog Authorship Attribution

Code related to tackle the problem of authorship attribution over micro blog social media like Twitter. The tag [`v1.2-tifs_2016`](https://github.com/theocjr/social-media-forensics/releases/tag/v1.2-tifs_2016) is the code used to perform the experiments described in the paper "Authorship Attribution for Social Media Forensics" to be published on IEEE Transactions on Information Forensics and Security. If you find this work relevant to your job please cite the paper. In spite of this README file and the documentation in the code try to explain most of what has been done, the paper further details the operations performed and their rationales.

The code is organized into these three directories:

* `dataset_pre_processing`: code that pre-processes the data before triggering the model learning. Examples of pre-processing are language filtering, data tagging and n-grams generation.
* `classification`: code to learn a model and run a classifier.
* `experiment_scripts`: examples of scripts to run the code in the above directories.
* `char-grams_analysis`: code and scripts related to analysis of different values for character grams.
* `utils`: utility code like scripts for installing required libraries.

Each Python program has a `-h/--help` command-line option that presents an usage explanation describing each option meaning. Most of these option have default values.


#### Dataset Pre-Processing

The `dataset_pre_processing` directory contains code to act over the dataset in order to prepare it to be run by code located in `classification` directory. All the code are optional except `ngrams_generator.py` that must be run before triggering the classifiers. The idea is to invoke these scripts in a pipeline version, with flexibility to choose which scripts to run with the desired options. In this pipeline, the output data of a step is the input data of the next (no operation is done over the input data in any code). Different invocation orders can be chosen but a common one is:

1. `filter_retweets_few_words.py`
2. `filter_language_by_tweet.py`
3. `tagging_irrelevant_data.py`
4. `ngrams_generator.py`

After running `ngrams_generator.py` the output is ready to be run by the classifiers presented in the `classification` directory.

Below is a list of each file/directory with a brief explanation and example of invocation when applicable (most of command-line options have default values):

* `messages_persistence.py`: Auxiliary code for manipulating (read and write) Twitter messages in dataset files.
* `guess-language-0.2`: Language detection API available in https://pypi.python.org/pypi/guess-language included in the project for convenience.
* `filter_retweets_few_words.py`: Code for reading authors' tweets files and remove retweets and tweets with few words.
  * Example of use: `./filter_retweets_few_words.py  --source-dir-data my_input_dir --dest-dir my_output_dir --filter-retweets --minimal-number-words 3 --debug`
* `filter_language_by_tweet.py`: Code for reading authors' tweets files and filter the messages based on an API of language detection.
  * Example of use: `./filter_language_by_tweet.py --source-dir my_input_dir --dest-dir my_output_dir --language-detection-module ./guess-language-0.2/guess_language/ --language English --debug`
* `tagging_irrelevant_data.py`: Code for tagging irrelevant data as numbers, dates, times, URLs, hashtags and user references. There are command-line options to individually suppress the tagging of each data type.
  * Example of use: `./tagging_irrelevant_data.py --source-dir my_input_dir --dest-dir my_output_dir --debug`
* `ngrams_generator.py`: Code for generating n-grams. The input are the messages presented in the dataset and the output are files in Python's built-in persistence model format implemented by `sklearn`, one file for each feature of each author (char-4-gram, word-1-gram, word-2-gram, ...) to be fed to the classifiers. This is the only code needed to be run before any classifier in the `classification` directory.
  * Example of use: `./ngrams_generator.py --source-dir my_input_dir --dest-dir my_output_dir --features all --debug`


#### Classification

The `classification` directory contains code related to the classifiers used to learn a model for the authors in the dataset. They expect that the dataset was previously pre-processed by the code in the `dataset_pre_processing` directory (at least `ngrams_generator.py` code) so that their input be the messages n-grams. Below is a list of the files/directories with brief explanations and examples of usage when applicable (most of command-line options have default values):

* `PmSVM`: Power Mean SVM classifier written by Jianxin Wu, originally available in https://sites.google.com/site/wujx2001/home/power-mean-svm. The original code was modified to deal with large data. The version history in this project can be used to see these modifications.
* `fest`: FEST software written by Nikos Karampatziakis that provides Random Forest classifiers. Available in http://lowrank.net/nikos/fest/ and included here for convenience.
* `pmsvm_classifier.py`: Program to trigger a Power Mean SVM classifier. Its input are the n-grams output by `ngrams_generator.py` code.
  * Example of use: `./pmsvm_classifier.py --source-dir-data my_input_dir --output-dir my_output_dir --minimal-number-tweets 1000 --validation-folding 10 --repetitions 10 --number-authors 50 --number-tweets 500 --features all --debug`
* `pmsvm_classifier_no_cross.py`: Program to trigger a Power Mean SVM classifier. Its input are the n-grams output by `ngrams_generator.py` code. This code is intended to be used in experiments with small datasets (less than 10 messages per author) so instead of a k-fold cross validation, a validation based on sampling is performed. First a test set is sampled and after several runs are conducted, each one with a different training set built by sampling. At the end a mean accuracy is calculated.
  * Example of use: `./pmsvm_classifier_no_cross.py --source-dir-data my_input_dir --output-dir my_output_dir --minimal-number-tweets 1000 --number-authors 50 --number-tweets 500 --features all --debug`
* `pmsvm_pca_classifier.py`: Program to trigger a Power Mean SVM (PmSVM) classifier. Its input are the n-grams output by `ngrams_generator.py` code. Before performing the PmSVM classification, a PCA decomposition is conducted in order to reduce the dimensionality.
 * Example of use: `./pmsvm_pca_classifier.py --source-dir-data my_input_dir --output-dir my_output_dir --minimal-number-tweets 1000 --validation-folding 10 --repetitions 10 --number-authors 50 --number-tweets 500 --features all --pca-variance 0.9 --debug`
* `rf_classifier.py`: Program to trigger a Random Forest (RF) classifier based on scikit-learn code. Its input are the n-grams output by `ngrams_generator.py` code. Besides doing the classification, it also calculates the importance of each feature used.
  * Example of use: `./rf_classifier.py --source-dir-data my_input_dir --output-dir my_output_dir --minimal-number-tweets 1000 --validation-folding 10 --repetitions 10 --number-authors 50 --number-tweets 500 --features all --number-trees 500 --number-most-important-features 100 --debug`
* `feature_vectors_generator.py`: Code to read n-grams data and to output feature vectors in libsvm format. Its input are the n-grams output by `ngrams_generator.py` code. This code is used to prepare the data for the `rf_classifier_fest.py` code (see below).
  * Example of use: `./feature_vectors_generator.py --source-dir-data my_input_dir --output-dir my_output_dir --minimal-number-tweets 1000 --validation-folding 10 --repetitions 10 --number-authors 50 --number-tweets 500 --features all --debug`
* `rf_classifier_fest.py`: Program to trigger a Random Forest classifier based on FEST code. Its input are the feature vectors output by `feature_vectors_generator.py` code. FEST code needs to be compiled as indicated in http://lowrank.net/nikos/fest/ before `rf_classifier_fest.py` invocation. The FEST code doesn't perform a multi-class classification (only binary) so a model for each author is learned.
  * Example of use: `./rf_classifier_fest.py --source-dir-data my_input_dir --output-dir my_output_dir --number-trees 500 -debug`


#### Experiment Scripts

The `experiment_scripts` directory contains lots of shell scripts used to run the experiments and serves as a container of invocation examples of the above code.


#### Char-grams Analysis

The `char-grams_analysis` directory has some of the code described above with modifications to deal with different values of character grams in order to perform an efficiency analysis.


#### Utils

The `utils` directory contains a single script called `install_libraries.sh` aimed to install all the dependencies on a fresh new Ubuntu Server LTS 64 14.04 (Trusty Tahr).


## License

The software in this project is distributed under [BSD License](http://www.linfo.org/bsdlicense.html).


Please, feel free to send any comments or suggestions for updates in these instructions.
