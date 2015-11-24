# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from utils import shared_zeros, shared_scalar
import pdb


class OPTIMIZER(object):
    """
    the basic class of optimizer
    """
    def __init__(self):
        self.gparams_acc = None
        self.n_acc = 0
    def gparams_update(self, gradients):
        """
        accumulate the iter gradients
        """

        #self.gparams_acc = gradients if self.gparams_acc == None else self.gparams_acc + gradients
        if self.gparams_acc == None:
            self.gparams_acc = gradients
        else:
            for i in xrange(len(gradients)):
                self.gparams_acc[i] += gradients[i]
        # TODO: DEBUG
        #print self.gparams_acc[6]
        self.n_acc += 1
        return


class SGD(OPTIMIZER):
    """
    """

    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.9,
                 decay=0.01):
        """
        """
        OPTIMIZER.__init__(self)
        self.updates ={}
        self.n_acc = 0
        self.delta_pre = {}
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        #self.lr_var = shared_scalar(self.learning_rate)
        #self.momentum_var = shared_scalar(momentum)

    def delta_pre_init(self, params):
        """
        """
        for param in params:
            self.delta_pre[param] = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                                           dtype=theano.config.floatX))
        return
    def params_update(self, params):
        """
        """
        for param, gparam in zip(params, self.gparams_acc):
            weight_update = self.delta_pre[param].get_value()
            temp = np.ones_like(gparam) * self.learning_rate / self.n_acc
            ugd = self.momentum * weight_update - gparam * temp
            #self.updates[weight_update] = ugd
            #self.updates[param] = param + ugd
            param.set_value(param.get_value() + ugd)
            self.delta_pre[param].set_value(ugd)

        # re-init
        self.gparams_acc = None
        #print params[6][-1].eval()
        self.n_acc = 0
        return
    def learning_rate_decay(self):
        #self.learning_rate *= (1-self.decay)
        #self.lr_var = shared_scalar(self.learning_rate)
        self.learning_rate *= (1-self.decay)
        return

class ADADELTA(OPTIMIZER):
    """
    the ada-delta optimizer appraoch
    """
    def __init__(self,
                params,
                learning_rate=1,
                decay=0.95,
                epsilon=1e-6):
        OPTIMIZER.__init__(self)
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.n_acc = 0
        self.acc_grad = {}
        self.acc_delta = {}
        for param in params:
            self.acc_grad[param] = np.zeros_like(param.get_value())
            self.acc_delta[param] = np.zeros_like(param.get_value())
            #self.acc_update[param] = np.zeros_like(param.get_value())
        return

    def params_update(self, params):
        for p, g_acc in zip(params, self.gparams_acc):
            g = (1. / self.n_acc) * np.ones_like(g_acc) * g_acc
            #g = g_acc
            self.acc_grad[p] = self.decay * self.acc_grad[p] + \
                               (1 - self.decay) * np.ones_like(g) * g * g
            ugd =  - np.sqrt(self.acc_delta[p] + self.epsilon) / \
                   np.sqrt(self.acc_grad[p] + self.epsilon) * g
            self.acc_delta[p] = self.decay * self.acc_delta[p] + \
                                (1-self.decay) * ugd**2
            p.set_value(p.get_value() + ugd)

        self.gparams_acc = None
        self.n_acc = 0
        return

    def learning_rate_decay(self):
        self.learning_rate *= self.decay
        return

    def numpy_floatX(self, data):
        return numpy.asarray(data, dtype=config.floatX)
