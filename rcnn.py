# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np


from vectorize import Vectorize
import data_process
import rnn
import cnn
import pdb
import logging
import sys


class RCNN(object):

    def __init__(self,
                 rng,
                 input_data,
                 dim,
                 n_feature_maps,
                 window_sizes,
                 n_hidden,
                 n_out,
                 h_prev,
                 y_prev):

        self.cnn = cnn.CNN(input_data=input_data,
                           rng=rng,
                           dim=dim,
                           n_feature_maps=n_feature_maps,
                           window_sizes=window_sizes)
        self.rnn = rnn.RNN(input_data=self.cnn.output,
                           rng=rng,
                           n_in = n_feature_maps*len(window_sizes),
                           n_hidden=n_hidden,
                           n_out=n_out,
                           h_prev=h_prev,
                           y_prev=y_prev,
                           activation=T.nnet.sigmoid)

        self.h = self.rnn.h
        self.window_sizes = window_sizes
        self.dim = dim
        self.n_out = n_out
        self.n_hidden = self.rnn.n_hidden
        self.params = self.cnn.params + self.rnn.params
        self.output = self.rnn.output
        self.loss = self.rnn.loss
        self.error = self.rnn.error
        return
