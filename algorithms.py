from __future__ import division

import random

import numpy as np
import sys

import utilities as utils


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class Kmeans(Classifier):

    def __init__(self, parameters={}):
        # Default: no of centers = 10
        self.params = {'nc':10}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)

    def learn(self, Xtrain, ytrain):

        ns = Xtrain.shape[0]                      # number of sample
        nc = self.params['nc']                    # number of centers
        pss = Xtrain[1].shape[0]                  # per sample shape
        c = utils.getRandCenters(Xtrain, nc, beta=np.ones(nc))[0]
        clusters = {}
        self.clusters, self.centroids, self.labels = [], [], []

        while True:

            # define clusters
            for cluster_id in range(nc):
                clusters[cluster_id] = []

            # assign samples to nearest clusters
            for sample_id in range(ns):
                dist_to_centriods = np.linalg.norm(np.tile(Xtrain[sample_id,], (nc,1))-c, axis=1)
                cluster_id = np.argmin(dist_to_centriods)
                clusters[cluster_id].append(sample_id)

            # recalibrate the weights
            old_c = c
            c = []
            centroid = {}
            for cluster_id, cluster in clusters.items():
                centroid[cluster_id] = np.mean(Xtrain[cluster], axis=0)
                c.append(centroid[cluster_id])

            # break the loop if oscillations are less than prefixed constant
            if np.sum(np.abs(np.subtract(c,old_c))) < 10:
                break

        for cluster_id, cluster in clusters.items():
            '''
            0 - ylabel
            1 - centriod
            '''
            if cluster == []:
                self.clusters.append(c[cluster_id])
                self.labels.append(ytrain[0])
                self.centroids.append(c[cluster_id])
                # self.cluster_labels_centroid.append([ytrain[0], c[cluster_id]])
            else:
                self.clusters.append(Xtrain[cluster])
                self.labels.append(utils.most_common(ytrain[cluster]))
                self.centroids.append(c[cluster_id])
                # self.cluster_labels_centroid.append([utils.most_common(ytrain[cluster]), c[cluster_id]])

        return self.centroids, self.clusters

    def predict(self, Xtest):

        P = []
        labels, centroids = self.labels, self.centroids
        # for cl in self.cluster_labels:
        #     labels.append(cl[0])
        #     centroids.append(cl[1])

        for sample in Xtest:
            replicated_sample = np.tile(sample, (len(labels),1))
            dist_to_centeriods = np.linalg.norm(replicated_sample - centroids, axis=1)
            selected_centriod_index = np.argmin(dist_to_centeriods)
            class_label = labels[selected_centriod_index]
            P.append(class_label)

        return P

class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        '''

        :param parameters: dictionary of various parameters
        '''
        self.params = {
            'hidden_nw_str': [30],
            'mbs': 10,
            'stepsize': 0.01,
            'transfer': 'sigmoid',
            'cost':'crossentropyloss',
            'epochs': 1,
            'regularization': None,
            'lmbda' : 0.1
        }
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)

        # activation function assingment
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> cannot handle your transfer function')

        # cost function assignment
        if self.params['cost'] == 'squareloss':
            self.dcost = utils.dsqloss
        elif self.params['cost'] == 'crossentropyloss':
            self.dcost = utils.dceloss
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> cannot handle loss function')

        # Regularization function assignment
        if self.params['regularization'] == None:
            self.reg = utils.reg
        elif self.params['regularization'] == 'L1':
            self.reg = utils.regl1
        elif self.params['regularization'] == 'L2':
            self.reg = utils.regl2
        else:
            # For now, only allowing L1 and L2 regularization
            raise Exception('NeuralNet -> cannot handle regularlization')

        self.w, self.b = [], []

    def _initRandW(self,nw_str,):
        w = [np.random.randn(y, x) for x,y in zip(nw_str[:-1],nw_str[1:])]
        b = [np.random.randn(y, 1) for y in nw_str[1:]]
        return w, b

    def __initZeroW(self):
        wzeros = [np.zeros(w.shape) for w in self.w]
        bzeros = [np.zeros(b.shape) for b in self.b]
        return wzeros, bzeros

    def learn(self, Xtrain, ytrain,):

        ni = Xtrain.shape[1]
        no = ytrain.shape[1]
        hidden_nw_str = self.params['hidden_nw_str']
        nw_str = [ni] + hidden_nw_str + [no]
        self.nts = len(Xtrain)              # number of training samples
        self.w, self.b = self._initRandW(nw_str)
        self.lw = len(self.w)
        self.SGD(Xtrain,ytrain)

    def SGD(self, Xtrain, ytrain,):
        epochs = self.params['epochs']      # number of passes on train data
        eeta = self.params['stepsize']      # step size
        mbs = self.params['mbs']            # mini batch size
        nts = self.nts                      # no of training samples
        lmbda = self.params['lmbda']
        n = self.nts
        Z = [i for i in range(nts)]


        for epoch in range(epochs):
            # print(epoch)
            random.shuffle(Z)
            Xtrain, ytrain = Xtrain[Z], ytrain[Z]

            # generating mini batches
            mini_batches = [Z[k:k+mbs] for k in range(0, nts, mbs)]
            # mini_batches = [training_data[k:k+mbs] for k in range(0, nts, mbs)]

            for mini_batch in mini_batches:
                mini_batch_data = zip(Xtrain[mini_batch], ytrain[mini_batch])
                w_batch_update, b_batch_update = self.mini_batch_update(mini_batch_data)

                # update the weights: Note the weights are being regularized where as the biases are not

                # print(self.w, self.b)

                self.w = [w - ((eeta/mbs)*w_change) + self.reg(w, lmbda, n, eeta)
                        for w, w_change in zip(self.w, w_batch_update)]
                self.b = [b - ((eeta/mbs)*b_change)
                        for b, b_change in zip(self.b, b_batch_update)]


    def mini_batch_update(self, mini_batch):

        w_batch_update, b_batch_update = self.__initZeroW()

        for x, y in mini_batch:

            w_sample_update, b_sample_update = self.backprop(x, y)

            for w in range(self.lw):
                # print(w_batch_update[w].shape, b_batch_update[w].shape)
                # print(w_sample_update[w].shape, b_sample_update[w].shape)
                w_batch_update[w] += w_sample_update[w]
                # print(b_batch_update[w].shape, b_sample_update[w].shape)
                b_batch_update[w] += b_sample_update[w]

        return w_batch_update, b_batch_update

    def backprop(self, x, y):

        x = np.atleast_2d(x).T
        y = np.atleast_2d(y).T

        w_sample_update, b_sample_update = self.__initZeroW()

        lli, llo = self.feedforword(x)

        # Backward pass Last Layer
        delta = self.dcost(llo[-1], y) * self.dtransfer(lli[-1])
        w_sample_update[-1] = np.dot(delta, llo[-2].T)
        b_sample_update[-1] = delta

        # Backward pass Other Layers

        for w in range(2, self.lw+1):
            delta = np.dot(self.w[-w + 1].T, delta) * self.dtransfer(lli[-w])
            w_sample_update[-w] = np.dot(delta, llo[-w - 1].T)
            b_sample_update[-w] = delta

        return (w_sample_update, b_sample_update)

    def feedforword(self, ip):
        '''
        function for calculating the weights across various layers
        :param inputs:
        :return:
        '''
        lli=[]  # layer input list (lli: list of layer inputs)
        llo=[]  # layer ouput list (llo: list of layer outputs)

        li = ip
        llo.append(li)

        for w,b in zip(self.w, self.b):
            # print(w.shape, llo[-1].shape, b.shape)
            lli.append(np.dot(w, llo[-1]) + b)
            llo.append(self.transfer(lli[-1]))

        return lli, llo

    def predict(self, Xtest):
        predictions = []
        for x in Xtest:
            xT = np.atleast_2d(x).T
            P = self.feedforword(xT)[-1][-1]
            p = np.argmax(P)
            predictions.append(p)
        predictions = np.array(predictions)
        return predictions
