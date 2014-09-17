# pylearn2 tutorial example: make_dataset.py by Ian Goodfellow
# See README before reading this file
#
#
# This script creates a preprocessed version of a dataset using pylearn2.
# It's not necessary to save preprocessed versions of your dataset to
# disk but this is an instructive example, because later we can show
# how to load your custom dataset in a yaml file.
#
# This is also a common use case because often you will want to preprocess
# your data once and then train several models on the preprocessed data.

import os.path
import pylearn2
from pylearn2.utils import serial
import readCSVforPylearn
from pylearn2.datasets import preprocessing

if __name__ == "__main__":
    train = readCSVforPylearn.CSVDataset(path = 'train.csv', start = 0, stop = 50000, one_hot = True, expect_headers = True)
    train_pkl_path = os.path.join('.', 'MNIST_train.pkl')
    serial.save(train_pkl_path, train)
    
    validate = readCSVforPylearn.CSVDataset(path = 'train.csv', start = 50000, stop = 60000, one_hot = True, expect_headers = True)
    validate_pkl_path = os.path.join('.', 'MNIST_validate.pkl')
    serial.save(validate_pkl_path, train)

    test = readCSVforPylearn.CSVDataset(path = 'test.csv', one_hot = True, expect_headers = True, expect_labels = False)
    test_pkl_path = os.path.join('.', 'MNIST_test.pkl')    
    serial.save(test_pkl_path, train)