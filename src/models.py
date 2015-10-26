# -*- coding: utf-8 -*-
import pdb
import theano.tensor as T
import rnn
import cnn
import utils

class RCNN_OneStep(object):
    """ The RCNN model with batch 1

    """

    def __init__(self,
                 input_var,
                 word_dim,
                 cnn_feature_maps,
                 cnn_window_sizes,
                 rnn_hidden,
                 rnn_output,
                 h_tm1):

        self.cnn_onestep_model = cnn.CNN_OneStep(input_var,
                                           word_dim,
                                           cnn_feature_maps,
                                           cnn_window_sizes)
        self.cnn_onestep_model.build_network()
        self.rnn_onestep_model = rnn.RNN_OneStep(self.cnn_onestep_model.output,
                                           cnn_feature_maps*len(cnn_window_sizes),
                                           rnn_hidden,
                                           rnn_output,
                                           h_tm1)
        self.params = self.cnn_onestep_model.params + self.rnn_onestep_model.params
        self.input_var = input_var
        self.h_pre_var = h_tm1
        self.word_dim = word_dim
        self.rnn_hidden = rnn_hidden
        self.rnn_onestep_model.build_network()
        self.h = self.rnn_onestep_model.h
        self.loss = self.rnn_onestep_model.loss
        self.y_pred = self.rnn_onestep_model.y_pred
        self.output = self.rnn_onestep_model.output
        return
