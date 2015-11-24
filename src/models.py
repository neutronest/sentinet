# -*- coding: utf-8 -*-
import pdb
import theano, theano.tensor as T
import rnn
import cnn
import utils
import layer


class Model(object):
    def __init__(self,
                 level1_model_name,
                 level2_model_name,
                 input_var,
                 lookup_table,
                 level1_input,
                 level1_hidden,
                 level2_input,
                 level2_hidden,
                 n_output,
                 word_dim=None,
                 n_feature_maps=None,
                 window_sizes=None,
                 if_dropout="dropout"):

        # variable init
        self.level1_model_name = level1_model_name
        self.level2_model_name = level2_model_name
        self.input_var = input_var
        self.level1_input = level1_input
        self.level1_hidden = level1_hidden
        self.level2_input = level2_input
        self.level2_hidden = level2_hidden
        self.n_output = n_output
        self.word_dim = word_dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes


        if word_dim is not None:
            self.word_dim = word_dim
        if n_feature_maps is not None:
            self.n_feature_maps = n_feature_maps
        if window_sizes is not None:
            self.window_sizes = window_sizes
        self.smodel = None
        self.tmodel = None

        if self.level1_model_name == "srnn_model":
            self.smodel = rnn.SRNN(input_var,
                                   lookup_table,
                                   self.level1_input,
                                   self.level1_hidden)
        elif self.level1_model_name == "sgru_model":
            self.smodel = rnn.SGRU(input_var,
                                   lookup_table,
                                   self.level1_input,
                                   self.level1_hidden)
        elif self.level1_model_name == "slstm_model":
            self.smodel = rnn.SLSTM(input_var,
                                    lookup_table,
                                    self.level1_input,
                                    self.level1_hidden)
        elif self.level1_model_name == "scnn_model":
            self.level2_input = len(self.window_sizes)*n_feature_maps
            self.smodel = cnn.SCNN(self.input_var,
                                   self.word_dim,
                                   self.n_feature_maps,
                                   self.window_sizes)
        else:
            print "smodel foobar 233333"
        self.smodel.build_network()

        if self.level2_model_name == "trnn_model":
            self.tmodel = rnn.TRNN(self.smodel.h,
                                   self.level2_input,
                                   self.level2_hidden,
                                   self.n_output)
        elif self.level2_model_name == "tgru_model":
            self.tmodel = rnn.TGRU(self.smodel.h,
                                   self.level2_input,
                                   self.level2_hidden,
                                   self.n_output)
        elif self.level2_model_name == "tlstm_model":
            self.tmodel = rnn.TLSTM(self.smodel.h,
                                    self.level2_input,
                                    self.level2_hidden,
                                    self.n_output)
        else:
            print "tmodel foorbar 23333"
        self.tmodel.build_network()

        self.lookup_table = self.smodel.lookup_table
        self.output_layer = layer.OutputLayer(n_output,
                                              self.tmodel.y,
                                              if_dropout)
        self.relation_pairs = self.tmodel.relation_pairs

        self.mask = self.smodel.mask
        self.h0 = self.smodel.h0
        self.th = self.tmodel.th
        if self.level2_model_name == "tlstm_model":
            self.c0 = self.smodel.c0
            self.tc = self.tmodel.tc
        else:
            # NEVER USED
            self.c0 = T.fmatrix('c0')
            self.tc = T.fvector('tc')

        self.y_pred = self.output_layer.y_pred
        self.y_drop_pred = self.output_layer.y_drop_pred
        self.output = self.output_layer.output
        self.loss = self.output_layer.loss
        self.error = self.output_layer.error
        self.params = self.smodel.params + self.tmodel.params
        self.params.append(self.lookup_table)

        # prepare L2_sqr
        self.L2 = 0
        for p in self.params:
            self.L2 += (p**2).sum()
        self.L2 = T.sqrt(self.L2) * 0.01
        return

""" BELOW ALL ABANDON!"""
class SRNN_TRNN(object):
    """
    the rnn-(tree rnn) model

    first using RNN to generate sentence vector,
    then appling the sentence vector to the tree rnn
    """
    def __init__(self,
                 input_var,
                 level1_input,
                 level1_hidden,
                 level2_input,
                 level2_hidden,
                 n_output):
        """
        the RNN-TRNN model
        input_var: the input variable
            type: theano tensor of dvector

        """

        self.input_var = input_var
        self.level1_input = level1_input
        self.level1_hidden = level1_hidden
        self.level2_input = level2_input
        self.level2_hidden = self.level2_hidden
        self.n_output = n_output

        self.srnn_model = rnn.SRNN(self.input_var,
                                   self.level1_input,
                                   self.level1_hidden)
        self.srnn_model.build_network()

        self.trnn_model = rnn.TRNN(self.srnn_model.h,
                                   self.level2_input,
                                   self.level2_hidden)
        self.trnn_model.build_network()

        self.sens_pos_var = self.srnn_model.sens_pos_var
        self.relation_pairs = self.trnn_model.relation_pairs
        self.th = self.trnn_model.th

        self.output_layer = layer.OutputLayer(level2_hidden,
                                               n_output,
                                               self.trnn_model.h,
                                               "uniform",
                                               "dropout")
        self.y_pred = self.output_layer.y_pred
        self.y_drop_pred = self.output_layer.y_drop_pred
        self.output = self.output_layer.output
        self.loss = self.output_layer.loss
        self.error = self.output_layer.error
        self.params = self.srnn_model.params + \
                      self.trnn_model.params + \
                      self.output_layer.params
        return

class SCNN_TRNN(object):
    """

    """
    def __init__(self,
                 input_var,
                 word_dim,
                 n_feature_maps,
                 window_sizes,
                 level2_input,
                 level2_hidden,
                 n_output):

        self.input_var = input_var
        self.word_dim = word_dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes
        self.level2_input = level2_input
        self.level2_hidden = level2_hidden
        self.n_output = n_output

        self.scnn_model = cnn.SCNN(self.input_var,
                                   self.word_dim,
                                   self.n_feature_maps,
                                   self.window_sizes)
        self.scnn_model.build_network()
        self.trnn_model = rnn.TRNN(self.scnn_model.output,
                                   self.level2_input,
                                   self.level2_hidden)
        self.trnn_model.build_network()
        self.output_layer = layer.OutputLayer(level2_hidden,
                                               n_output,
                                               self.trnn_model.h,
                                               "uniform",
                                               "dropout")
        self.y_pred = self.output_layer.y_pred
        self.y_drop_pred = self.output_layer.y_drop_pred
        self.output = self.output_layer.output
        self.loss = self.output_layer.loss
        self.error = self.output_layer.error
        self.params = self.scnn_model.params + \
                      self.trnn_model.params + \
                      self.output_layer.params
        return


class SGRU_TGRU(object):
    """

    """
    def __init__(self,
                 input_var,
                 level1_input,
                 level1_hidden,
                 level2_input,
                 level2_hidden,
                 n_output):
        """
        the SGRU-TGRU init

        """
        self.input_var = input_var
        self.level1_input = level1_input
        self.level1_hidden = level1_hidden
        self.level2_input = level2_input
        self.level2_hidden = level2_hidden

        self.sgru_model = rnn.SGRU(self.input_var,
                                   self.level1_input,
                                   self.level1_hidden)
        self.sgru_model.build_network()

        self.tgru_model = rnn.TGRU(self.sgru_model.hidden_states_var,
                                   self.level2_input,
                                   self.level2_hidden)
        self.tgru_model.build_network()
        self.sens_pos_var = self.sgru_model.sens_pos_var
        self.relation_pairs = self.tgru_model.relation_pairs
        self.th = self.tgru_model.th

        self.output_layer = layer.OutputLayer(level2_hidden,
                                               n_output,
                                               self.tgru_model.h,
                                               "uniform",
                                               "dropout")
        self.y_pred = self.output_layer.y_pred
        self.y_drop_pred = self.output_layer.y_drop_pred
        self.output = self.output_layer.output
        self.loss = self.output_layer.loss
        self.error = self.output_layer.error
        self.params = self.sgru_model.params + \
                      self.tgru_model.params + \
                      self.output_layer.params
        return

class SLSTM_TLSTM(object):
    """
    """
    def __init__(self,
                 input_var,
                 level1_input,
                 level1_hidden,
                 level2_input,
                 level2_hidden,
                 n_output):
        """
        The SLSTM-TLSTM init
        """
        self.input_var = input_var
        self.level1_input = level1_input
        self.level1_hidden = level1_hidden
        self.level2_input = level2_input
        self.level2_hidden = level2_hidden
        self.n_output = n_output

        self.slstm_model = rnn.SLSTM(self.input_var,
                                     self.level1_input,
                                     self.level1_hidden)
        self.slstm_model.build_network()

        self.tlstm_model = rnn.TLSTM(self.slstm_model.h,
                                     self.level2_input,
                                     self.level2_hidden,
                                     self.n_output)
        self.tlstm_model.build_network()
        self.sens_pos_var = self.slstm_model.sens_pos_var
        self.relation_pairs = self.tlstm_model.relation_pairs
        self.th = self.tlstm_model.th
        self.tc = self.tlstm_model.tc

        self.output_layer = layer.OutputLayer(self.level2_hidden,
                                               self.n_output,
                                               self.tlstm_model.h,
                                               "uniform",
                                               "dropout")
        self.y_pred = self.output_layer.y_pred
        self.y_drop_pred = self.output_layer.y_drop_pred
        self.output = self.output_layer.output
        self.loss = self.output_layer.loss
        self.error = self.output_layer.error
        self.params = self.slstm_model.params + \
                      self.tlstm_model.params + \
                      self.output_layer.params
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
