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
        input_data: symbolic sentence tensor, a 3-dimension tensor
        type of input_data: theano.tensor 3d tensor

        dim: the dimensions of word vector
        type of dim: int

        n_feature_maps: number of feature maps
        type of n_feature_maps: int

        window_size: the filters
        type of window_size: tuple of int


        """
        self.input_data = input_data
        #self.input_data = input_data.dimshuffle('x', 'x', 0, 1)
        self.dim = dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes

        # params init
        self.params = []
        self.W_list = []
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
            self.W_list.append(W)
            self.params.append(W)
        b_init = np.asarray(np.zeros((self.n_feature_maps * len(self.window_sizes),), dtype=theano.config.floatX))
        self.b = theano.shared(value=b_init)
        self.params.append(self.b)


        def _conv(word_vector):
            word_matrix = word_vector.dimshuffle('x', 'x', 0, 1)
            self.h = None

            for i in xrange(len(window_sizes)):
                ws = window_sizes[i]
                conv_out = conv.conv2d(input=word_matrix, filters=self.W_list[i])
                max_out = T.max(conv_out, axis=2).flatten()
                self.h = max_out if self.h == None else \
                         T.concatenate([self.h, max_out])
                self.o = self.h + self.b
                return self.o

        [self.output], _ = theano.scan(_conv,
                                       sequences=self.input_data,
                                       outputs_info=[None])

        return
