# -*- coding: utf-8 -*-
import pdb
import theano, theano.tensor as T
import rnn
import cnn
import utils


class SRNN_TRNN(object):
    """
    the rnn-(tree rnn) model

    first using RNN to generate sentence vector,
    then appling the sentence vector to the tree rnn
    """
    def __init__(self,
                 input_var,
                 srnn_input,
                 srnn_hidden,
                 srnn_output,
                 trnn_input,
                 trnn_hidden,
                 trnn_output):
        """
        the RNN-TRNN model

        input_var: the input variable
            type: theano tensor of dvector

        """

        self.input_var = input_var
        self.srnn_input = srnn_input
        self.srnn_hidden = srnn_hidden
        self.srnn_output = srnn_output # abandon
        self.trnn_input = trnn_input
        self.trnn_hidden = trnn_hidden
        self.trnn_output = trnn_output

        # trick
        self.level2_hidden = trnn_hidden

        self.srnn_model = rnn.SRNN(self.input_var,
                                   self.srnn_input,
                                   self.srnn_hidden,
                                   self.srnn_output)
        self.srnn_model.build_network()

        self.trnn_model = rnn.TRNN(self.srnn_model.hidden_states_var,
                                   self.trnn_input,
                                   self.trnn_hidden,
                                   self.trnn_output)
        self.trnn_model.build_network()

        self.sens_pos_var = self.srnn_model.sens_pos_var
        self.relation_pairs = self.trnn_model.relation_pairs
        self.th = self.trnn_model.th
        self.y_pred = self.trnn_model.y_pred
        self.output = self.trnn_model.output
        self.loss = self.trnn_model.loss
        self.error = self.trnn_model.error
        self.params = [self.srnn_model.W_input,
                       self.srnn_model.W_hidden,
                       self.srnn_model.b_h,
                       self.srnn_model.h0,
                       self.trnn_model.TW_input,
                       self.trnn_model.TW_hidden,
                       self.trnn_model.TW_output,
                       self.trnn_model.tb_h,
                       self.trnn_model.tb_y]
        return

class SGRU_TGRU(object):
    """

    """
    def __init__(self,
                 input_var,
                 sgru_input,
                 sgru_hidden,
                 sgru_output,
                 tgru_input,
                 tgru_hidden,
                 tgru_output):
        """
        the SGRU-TGRU init

        """
        self.input_var = input_var
        self.sgru_input = sgru_input
        self.sgru_hidden = sgru_hidden
        self.sgru_output = sgru_output
        self.tgru_input = tgru_input
        self.tgru_hidden = tgru_hidden
        self.tgru_output = tgru_output

        # trick
        self.level2_hidden = tgru_hidden

        self.sgru_model = rnn.SGRU(self.input_var,
                                   self.sgru_input,
                                   self.sgru_hidden,
                                   self.sgru_output)
        self.sgru_model.build_network()

        self.tgru_model = rnn.TGRU(self.sgru_model.hidden_states_var,
                                   self.tgru_input,
                                   self.tgru_hidden,
                                   self.tgru_output)
        self.tgru_model.build_network()
        self.sens_pos_var = self.sgru_model.sens_pos_var
        self.relation_pairs = self.tgru_model.relation_pairs
        self.th = self.tgru_model.th
        self.y_pred = self.tgru_model.y_pred
        self.output = self.tgru_model.output
        self.loss = self.tgru_model.loss
        self.error = self.tgru_model.error
        self.params = self.sgru_model.params + self.tgru_model.params
        return





class RCNN_OneStep(object):
    """ The RCNN model with batch 1

    """

    def __init__(self,
                 input_var,
                 y_pre_var,
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
                                                 y_pre_var,
                                                 cnn_feature_maps*len(cnn_window_sizes),
                                                 rnn_hidden,
                                                 rnn_output,
                                                 h_tm1)
        self.params = self.cnn_onestep_model.params + self.rnn_onestep_model.params
        self.input_var = input_var
        self.y_pre_var = y_pre_var
        self.h_pre_var = h_tm1
        self.word_dim = word_dim
        self.rnn_hidden = rnn_hidden
        self.rnn_onestep_model.build_network()
        self.h = self.rnn_onestep_model.h
        self.loss = self.rnn_onestep_model.loss
        self.error = self.rnn_onestep_model.error
        self.y_pred = self.rnn_onestep_model.y_pred
        self.output_var = self.rnn_onestep_model.output_var
        return
