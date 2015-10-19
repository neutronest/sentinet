# -*- coding: utf-8 -*-
import pdb
import theano.tensor as T
import rnn
import cnn
import utils

class RCNN(object):
    """
    The RNN with CNN word embedding model
    """
    def __init__(self,
                 rng,
                 input_data,
                 dim,
                 n_feature_maps,
                 window_sizes,
                 n_hidden,
                 n_out
                ):
        self.cnn = cnn.CNN(input_data=input_data,
                           rng=rng,
                           dim=dim,
                           n_feature_maps=n_feature_maps,
                           window_sizes=window_sizes)
        self.rnn = rnn.RNN(input_data=self.cnn.output,
                           rng=rng,
                           n_input = n_feature_maps*len(window_sizes),
                           n_hidden=n_hidden,
                           n_output=n_out,
                           activation=T.nnet.sigmoid,
                           output_type="softmax")

        self.window_sizes = window_sizes
        self.dim = dim
        self.n_out = n_out
        self.n_hidden = self.rnn.n_hidden
        self.params = self.cnn.params + self.rnn.params
        self.output_var = self.rnn.output_var
        self.y = self.rnn.p_y_given_x_var
        self.loss = self.rnn.loss
        self.error = self.rnn.error
        return
