from __future__ import division  # floating point division
import numpy as np
import math

import time

import dataloader as dtl
import algorithms as algos


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

if __name__ == '__main__':

    numruns = 1
    learnername = 'NeuralNets'

    classalgs = {
        'Neural Network': algos.NeuralNet({'epochs': 100})
    }

    '''
    hidden_nw_str : includes hidden layers count of nodes.
        ex: [20, 30, 50] --> hidden layer1 20 neurons, hidden layer2 30 neurons and hidden layer3 50 neurons
    mbs : mini batch size
    transfer : activation function in the neural network
        currently available : [sigmoid]
    cost : cost function in the layer to compute the error
        currently available : [squared loss]
    epochs : number of times the entired data is used
    regularization : to avoid over fitting
        currently available : ['L2']
    '''
    parameters = (
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost':'squareloss',
        #  'epochs': 1, 'regularization': 'L2', 'lmbda':0.1},
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
        #  'epochs': 5, 'regularization': 'L2', 'lmbda':0.01},
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
        #  'epochs': 10, 'regularization': 'L2', 'lmbda':0.1},
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
        #  'epochs': 15, 'regularization': 'L2', 'lmbda': 0.1},
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
        #  'epochs': 30, 'regularization': 'L2', 'lmbda': 0.1},
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
        #  'epochs': 50, 'regularization': 'L2', 'lmbda': 0.1},
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
        #  'epochs': 80, 'regularization': 'L2', 'lmbda': 0.1},
        # {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
        #  'epochs': 100, 'regularization': 'L2', 'lmbda': 0.1},
        {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
         'epochs': 150, 'regularization': 'L2', 'lmbda': 0.1},
        {'hidden_nw_str': [100], 'mbs': 10, 'stepsize': 0.1, 'transfer': 'sigmoid', 'cost': 'crossentropyloss',
         'epochs': 300, 'regularization': 'L2', 'lmbda': 0.1},
    )
    numparams = len(parameters)

    errors = {}

    validationset_size = 1
    trainset, validationset, testset = dtl.load_mnist(validationset_size)
    # trainset, testset = dtl.load_TT()

    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):

        # print(
        #     ('Running on train={0} ,validation={1} and test={2} samples for run {2}').
        #         format(len(trainset[0]), len(validationset[0]), len(testset[0]), r))

        for p in range(numparams):
            params = parameters[p]

            for learnername, learner in classalgs.items():

                # Reset learner for new parameters
                learner.reset(params)
                print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                start_time = time.time()
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                end_time = time.time()
                print('Time Taken: ', end_time - start_time)
                error = geterror(testset[1], predictions)
                print('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p, r] = error


    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(
            1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))
