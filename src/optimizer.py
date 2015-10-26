# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from utils import shared_zeros, shared_scalar


class SGD(object):
    """
    """

    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.,
                 decay=0.99):
        """
        """

        self.updates = []
        self.gparams_acc = None
        self.lr_var = shared_scalar(learning_rate)
        self.momentum_var = shared_scalar(momentum)


    def gparams_update(self, gparams):

        if self.gparams_acc == None:
            self.gparams_acc = gparams
        else:
            self.gparams_acc = [gparam_acc + gparam for (gparam_acc, gparam) in zip(self.gparams_acc, gparams)]
        return self.gparams_acc

    def param_update(self, param_var, gparam_var):
        ugd = - gparam_var * self.lr_var
        param_var += ugd
        return param_var


    def get_gradients(self, loss_var, params_var_list):
        """
        calculate the gradient of current data first
        do not update the param immediately

        Parameters:
        -----------
        loss_var: the loss variable about this time cost
          type: theano variable

        params_var_list: the list of parameters variable
          type: list of theano variable
        """
        gparams = [T.grad(loss_var, param_var) for param_var in params_var_list]
        return gparams

    def add_gradients(self, gparams):
        """
        gradients accumulate

        Parameters:
        -----------
        gparams: the gradients that need to be add
          type: list of theano variable

        """
        self.gparams_acc = [g_acc + g for g_acc, g in zip(gparams_acc, gparams)]
        return

    def update_params(self, params_var_list):
        """
        """
        for params_var, gparam in zip(params_var_list, self.gparams_acc):
            ugd = - gparam * self.lr_var
            params_var += ugd
        return

    def update_params_momentum(self,
                               params_var_list,
                               momentum,
                               pre_ugd_list):
        for params_var, gparam, pre_ugd in zip(params_var_list, self.gparams_acc, pre_ugd_list):
            ugd = - gparam * self.lr_var
            params_var += momentum * pre_ugd + ugd
        return

    def clear_gradients(self):
        """
        """
        self.gparams_acc = []
        return
