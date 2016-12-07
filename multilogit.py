import math
import numpy as np
import dataloader as dtl
import algorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct / float(len(ytest))) * 100.0


def geterror(ytest, predictions):
    return (100.0 - getaccuracy(ytest, predictions))


# if __name__ == '__main__':
def start():

    numruns = 30

    trainset, validationset, testset = dtl.load_mnist(validation_size=1)

    classalgs = {
        'Multinomial Logistic Regression': algs.MultiLogitReg(),
    }
    numalgs = len(classalgs)

    parameters = (
        {'stepsize': 0.1, 'mbs':10, 'epochs': 20, 'regularizer': 'L2', 'regwt': 0.1},
        # {'stepsize': 0.1, 'mbs':10, 'epochs': 10, 'regularizer': 'L1', 'regwt': 0.1},
        # {'stepsize': 0.1, 'mbs':10, 'epochs': 10, 'regularizer': None, 'regwt': 0.1},
        # {'stepsize': 0.1, 'mbs':10, 'epochs': 10, 'regularizer': None, 'regwt': 0.1},
    )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams, numruns))

    for r in range(numruns):

        print(
            ('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],
                                                                             r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print('Error for ' + learnername + ': ' + str(error))
                # print(r+1,',',error)
                errors[learnername][p, r] = error

    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0, :])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p, :])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(
            1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))

    return errors[learnername][0]