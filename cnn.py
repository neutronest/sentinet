## -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np

class CNN(object):

    def __init__(self,
                input_data,
                rng,
                dim,
                n_feature_maps,
                window_sizes):
        """
        Params:
        -------
        input_data: symbolic sentence tensor
        type of input_data: theano.tensor.matrix

        dim: the dimensions of word vector
        type of dim: int

        n_feature_maps: number of feature maps
        type of n_feature_maps: int

        window_size: the filters
        type of window_size: tuple of int


        """
        self.input_data = input_data.dimshuffle('x', 'x', 0, 1)
        self.dim = dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes
        self.params = []

        self.h = None
        for ws in window_sizes:
            # ws declare each window_size
            W_init = np.asarray(rng.uniform(low=-0.1,
                                       high=0.1,
                                       size=(self.n_feature_maps,
                                             1,
                                             ws,
                                             self.dim)),
                           dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name="W")
            self.params.append(W)

            conv_out = conv.conv2d(input=self.input_data, filters=W)
            max_out = T.max(conv_out, axis=2).flatten()


            self.h = max_out if self.h == None else \
                          T.concatenate([self.h, max_out])
        b_init = np.asarray(np.zeros((self.n_feature_maps * len(self.window_sizes),), dtype=theano.config.floatX))
        self.b = theano.shared(value=b_init)
        self.params.append(self.b)
        self.output = self.h + self.b
        return
