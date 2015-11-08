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
        self.decay = decay
        self.lr_var = shared_scalar(self.learning_rate)
        self.momentum_var = shared_scalar(momentum)

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
            weight_update = self.delta_pre[param]
            ugd = self.momentum_var * weight_update - gparam * self.lr_var / self.n_acc
            self.updates[weight_update] = ugd
            self.updates[param] = param + ugd

        f = theano.function(inputs=[],
                            outputs=None,
                            updates=self.updates)
        f()
        """
        for i in xrange(len(params)):
            ugd = self.momentum_var * self.delta_pre[i] - self.gparams_acc[i] / self.n_acc * self.lr_var
            self.delta_pre[i] = ugd
            self.updates[i] = params[i] + ugd
        """
        # re-init
        self.gparams_acc = None
        #print params[6][-1].eval()
        self.n_acc = 0
        return
    def learning_rate_decay(self):
        #self.learning_rate *= (1-self.decay)
        #self.lr_var = shared_scalar(self.learning_rate)
        self.lr_var *= (1 - self.decay)
        return
