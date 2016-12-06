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
        'K means': algos.Kmeans()
    }


    parameters = (
        # {'nc':10},{'nc':20},{'nc':30},{'nc':40},
        # {'nc':50},{'nc':60},{'nc':70},{'nc':80},{'nc':90},
        {'nc':90},
    )
    numparams = len(parameters)

    errors = {}

    validationset_size = 1
    trainset, validationset, testset = dtl.load_mnist_kmeans(validationset_size)
    # trainset, testset = dtl.load_TT()

    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):

        print(
            ('Running on train={0} ,validation={1} and test={2} samples for run {2}').
                format(len(trainset[0]), len(validationset[0]), len(testset[0]), r))

        for p in range(numparams):
            params = parameters[p]

            for learnername, learner in classalgs.items():

                # Reset learner for new parameters
                learner.reset(params)
                print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                start_time = time.time()
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                end_time = time.time()
                print('Time Taken: ',end_time-start_time)
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
