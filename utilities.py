from __future__ import division  # floating point division
import math
import random
import itertools
import numpy as np
import operator


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def calculateprob(x, mean, stdev):
    if stdev < 1e-3:
        if math.fabs(x - mean) < 1e-2:
            return 1.0
        else:
            return 0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def sigmoid(xvec):
    """ Compute the sigmoid function """
    # Cap -xvec, to avoid overflow
    # Undeflow is okay, since it get set to zero
    # xvec[xvec < -100] = -100
    vecsig = 1.0 / (1.0 + np.exp(-xvec))

    return vecsig


def dsigmoid(xvec):
    """ Gradient of standard sigmoid 1/(1+e^-x) """
    vecsig = sigmoid(xvec)
    return vecsig * (1 - vecsig)


def l2(vec):
    """ l2 norm on a vector """
    return np.linalg.norm(vec)


def dl2(vec):
    """ Gradient of l2 norm on a vector """
    return vec


def l1(vec):
    """ l1 norm on a vector """
    return np.linalg.norm(vec, ord=1)


def dl1(vec):
    """ Subgradient of l1 norm on a vector """
    grad = np.sign(vec)
    grad[abs(vec) < 1e-4] = 0.0
    return grad


def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs), )
    classes[probs < 0.5] = 0
    return classes


def logsumexp(a):
    """
    Compute the log of the sum of exponentials of input elements.
    Modified scipys logsumpexp implemenation for this specific situation
    """

    awithzero = np.hstack((a, np.zeros((len(a), 1))))
    maxvals = np.amax(awithzero, axis=1)
    aminusmax = np.exp((awithzero.transpose() - maxvals).transpose())

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(aminusmax, axis=1))

    out = np.add(out, maxvals)

    return out


def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    for k in dict1:
        if k in dict2:
            dict1[k] = dict2[k]


def learndistribution(nparray):
    mu, sigma = np.mean(nparray), np.std(nparray, ddof=1)
    return mu, sigma


def sqrt_one_plus_xwSquare(xw):
    return np.sqrt(1 + np.square(xw))


def one_plus_xwSquare(xw):
    return (1 + np.square(xw))


def proximalOperator(w, metaparam, n, eeta):
    constant = (metaparam * eeta)
    w[w > constant] = w[w > constant] - constant
    w[w < constant] = w[w < constant] + constant
    w[np.absolute(w) < constant] = 0
    return w


def np_get_accuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if np.array_equal(ytest[i], predictions[i]):
            correct += 1
    return (correct / ytest.shape[0]) * 100.0


# Loss Functions

def dsqloss(prediction, actual):
    # squared Loss
    return prediction - actual


def dceloss(yhat, y):
    # derivative of Cross Entropy loss
    dloss = -((y / yhat) - ((1 - y) / (1 - yhat)))
    return dloss

# Regularizations Functions

def reg(w, lmbda, n, eeta):
    # No Regularization
    return 0


def regl1(w, lmbda, n, eeta):
    # L1 Regularization
    return proximalOperator(w, lmbda, n, eeta)


def regl2(w, lmbda, n, eeta):
    # L2 Regualarlization
    return eeta * (lmbda / n) * w

def rmBias(ip):
    var = np.var(ip, axis=0)
    biases = np.where(var == 0)
    unbiased = [i for i in range(ip.shape[1])]
    unbiased.remove(biases[0])
    return ip[:, unbiased]

def addBias(ip):
    ns = ip.shape[0]
    ones = np.atleast_2d(np.ones(ns)).T
    return np.append(ip, ones, axis=1)

def most_common(L):
    #
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key= operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

def getRandCenters(Xtrain, nc, beta):
    '''
    Generates random centers in the given dataset
    :param Xtrain: given data set
    :param nc: number of center to generate
    :param beta: bandwidth
    :return: random centers
    '''
    ns = Xtrain.shape[0]                                                  # number of sample
    rand_sample_indices = random.sample(range(0, ns), nc)
    rand_centers = Xtrain[rand_sample_indices]
    rand_variance = np.ones(nc)*beta
    return rand_centers, rand_variance

def getKmeanCenters(Xtrain, ytrain, nc, var=False):
    '''
    Generates centers in the given dataset using the k-means algorithm
    :param Xtrain: given data set
    :param nc: number of center to generate
    :return: random centers
    '''
    learner = algs.Kmeans()
    params = {'nc':nc}
    learner.reset(parameters=params)
    centroids, clusters = learner.learn(Xtrain, ytrain)

    variances = []

    if var==True:
        for cluster in clusters:
            variances.append(np.var(cluster, axis=0))

    return np.array(centroids), np.array(variances)
