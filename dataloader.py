from __future__ import division  # floating point division

import gzip
import pickle
import math
import numpy as np
import copy

import sys
from sklearn.preprocessing import normalize


####### Main load functions

def load_kaggle_digits():
    # digit recognization " Kaggle data set"
    # filenames = ['./data/train.csv', './data/test.csv']
    # size = 37000
    filenames = ['./data/sampleTrain.csv', './data/sampleTest.csv']
    size = 900
    train_load, test_load = loadcsv(filenames[0]), loadcsv(filenames[1])

    Xtrain = [np.reshape(x[1:], (784, 1)) / 255 for x in train_load[1:size]]
    ytrain = [vector(x[0]) for x in train_load[1:size]]

    Xtest = [np.reshape(x[1:], (784, 1)) / 255 for x in train_load[size:]]

    trainset = [Xtrain] + [ytrain]
    testset = [Xtest] + []
    # data_pickle(trainset, testset) # not working
    return trainset, testset


def load_mnist(validation_size=10):
    '''
    Data is from mnist data set that is converted into a csv format
    :return: tuple containing training set, validation set, test set
    '''
    filenames = ['./data/mnist_train.csv', './data/mnist_test.csv']
    # filenames = ['./data/sample_mnist_train.csv', './data/sample_mnist_test.csv']
    train_load = np.loadtxt(fname=filenames[0], delimiter=',')
    test_load = np.loadtxt(fname=filenames[1], delimiter=',')

    Xtrain = train_load[:-validation_size,1:]/255
    ytrain = np.zeros([Xtrain.shape[0],10])
    for y,yvec in zip(train_load[:-validation_size,0],ytrain):
        yvec[int(y)] = 1

    Xvalidate = train_load[-validation_size:, 1:]/255
    yvalidate = train_load[-validation_size:,0]

    Xtest = test_load[:, 1:]/255
    ytest = test_load[:, 0]


    trainset = [Xtrain] + [ytrain]
    validationset = [Xvalidate] + [yvalidate]
    testset = [Xtest] + [ytest]

    # print(Xtrain.shape,ytrain.shape)
    # print(Xvalidate.shape, yvalidate.shape)
    # print(Xtest.shape, ytest.shape)
    print('data loaded')
    return trainset, validationset, testset


def load_mnist_kmeans(validation_size=10):
    '''
    Data is from mnist data set that is converted into a csv format
    :return: tuple containing training set, validation set, test set
    '''
    filenames = ['./data/mnist_train.csv', './data/mnist_test.csv']
    # filenames = ['./data/sample_mnist_train.csv', './data/sample_mnist_test.csv']
    train_load = np.loadtxt(fname=filenames[0], delimiter=',')
    test_load = np.loadtxt(fname=filenames[1], delimiter=',')

    Xtrain = train_load[:-validation_size,1:]/255
    ytrain = train_load[:-validation_size,0]

    Xvalidate = train_load[-validation_size:, 1:]/255
    yvalidate = train_load[-validation_size:,0]

    Xtest = test_load[:, 1:]/255
    ytest = test_load[:, 0]

    trainset = [Xtrain] + [ytrain]
    validationset = [Xvalidate] + [yvalidate]
    testset = [Xtest] + [ytest]

    # print(Xtrain.shape,ytrain.shape)
    # print(Xvalidate.shape, yvalidate.shape)
    # print(Xtest.shape, ytest.shape)
    print('data loaded')
    return trainset, validationset, testset


def load_mnist_pickle():
    return unpickle('load_mnist_pickle')


def load_TT():
    Xtrain = np.atleast_2d([np.array([i, j, k,]) for i in [0, 1] for j in [0, 1] for k in [0, 1]])
    Xtest = np.vstack((Xtrain[1:,:],Xtrain[0,:].T))
    # Xtrain = np.hstack((Xtrain,np.atleast_2d(np.ones(Xtrain.shape[0])).T))
    # Xtrain = np.array([[i, j] for i in [0, 1] for j in [0, 1]])
    # Xtest = np.atleast_2d([0,1,1,0]).T
    trainset = []
    trainset.append(Xtrain)
    trainset.append(Xtest)
    testset = trainset
    return trainset, testset


####### Helper functions

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset


def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    randindices = np.random.randint(0, dataset.shape[0], trainsize + testsize)
    featureend = dataset.shape[1] - 1
    outputlocation = featureend
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize], featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize], outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize + testsize], featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize + testsize], outputlocation]

    if testdataset is not None:
        Xtest = dataset[:, featureoffset:featureend]
        ytest = dataset[:, outputlocation]

        # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:, ii]))
        if maxval > 0:
            Xtrain[:, ii] = np.divide(Xtrain[:, ii], maxval)
            Xtest[:, ii] = np.divide(Xtest[:, ii], maxval)

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0], 1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0], 1))))

    return ((Xtrain, ytrain), (Xtest, ytest))


def create_susy_dataset(filenamein, filenameout, maxsamples=100000):
    dataset = np.genfromtxt(filenamein, delimiter=',')
    y = dataset[0:maxsamples, 0]
    X = dataset[0:maxsamples, 1:9]
    data = np.column_stack((X, y))

    np.savetxt(filenameout, data, delimiter=",")


def make_pickle(file, filename):
    fileopen = open('data_pickle/' + filename, 'wb')
    pickle.dump(file, fileopen)
    fileopen.close()
    print('pickled')


def unpickle(filename):
    fileopen = open('data_pickle/' + filename, 'rb')
    unpickled = pickle.load(fileopen)
    fileopen.close()
    print('unpickled')
    return unpickled


def vector(y):
    vec = np.zeros((10, 1))
    vec[int(y)] = 1
    return vec