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

        if self.level1_model_name == "srnn_Model":
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
        elif self.level1_model_name == "slstm_avg_model":
            self.smodel = rnn.SLSTM_avg(input_var,
                                        lookup_table,
                                        self.level1_input,
                                        self.level1_hidden)
        elif self.level1_model_name == "cnn_model":
            self.smodel = cnn.CNN(input_var,
                                  lookup_table,
                                  word_dim,
                                  n_feature_maps,
                                  window_sizes,
                                  n_output,
                                  False)
            self.level2_input = len(self.window_sizes)*n_feature_maps

            # n_output never used
        else:
            print "none smodel"
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

        elif self.level2_model_name == "tlstm_s_model":
            self.tmodel = rnn.TLSTM_s(self.smodel.h,
                                      self.level2_input,
                                      self.level2_hidden,
                                      self.n_output)
        elif self.level2_model_name == "tlstm_f_model":
            self.tmodel = rnn.TLSTM_f(self.smodel.h,
                                      self.level2_input,
                                      self.level2_hidden,
                                      self.n_output)
        elif self.level2_model_name == "tlstm_fc_model":
            self.tmodel = rnn.TLSTM_fc(self.smodel.h,
                                       self.level2_input,
                                       self.level2_hidden,
                                       self.n_output)
        elif self.level2_model_name == "tlstm_fy_model":
            self.tmodel = rnn.TLSTM_fy(self.smodel.h,
                                       self.level2_input,
                                       self.level2_hidden,
                                       self.n_output)
        else:
            print "none tmodel"
        self.tmodel.build_network()

        self.lookup_table = self.smodel.lookup_table
        self.output_layer = layer.OutputLayer(n_output,
                                              self.tmodel.y,
                                              if_dropout)
        self.relations = self.tmodel.relations
        if self.level1_model_name != "cnn_model":
            self.mask = self.smodel.mask
            self.h0 = self.smodel.h0
        self.th = self.tmodel.th
        self.c0  = getattr(self.smodel, 'c0', T.fmatrix('c0'))
        self.tc = getattr(self.tmodel, 'tc', T.fvector('tc'))
        self.dt = getattr(self.tmodel, 'dt', T.fmatrix('dt'))
        self.yt = getattr(self.tmodel, 'yt', T.fmatrix('yt'))
        self.yt_pred = getattr(self.tmodel, 'yt_pred', T.fmatrix('yt_pred'))
        self.if_train_var = getattr(self.tmodel, 'if_train_var', T.scalar('if_train_var'))

        self.y_pred = self.output_layer.y_pred
        self.y_drop_pred = self.output_layer.y_drop_pred
        self.output = self.output_layer.output
        self.loss = self.output_layer.loss
        self.error = self.output_layer.error
        self.params = self.smodel.params + self.tmodel.params

        # prepare L2_sqr
        self.L2 = 0
        for p in self.params:
            self.L2 += (p**2).sum()
        self.L2 = T.sqrt(self.L2) * 0.01
        return

class SingleModel(object):
    """
    """
    def __init__(self,
                 model_name,
                 input_var,
                 lookup_table,
                 n_input,
                 n_hidden,
                 n_output,
                 if_dropout="dropout",
                 word_dim=None,
                 cnn_n_feature_maps=None,
                 cnn_window_sizes=None):
        self.model_name = model_name
        self.input_var = input_var
        self.lookup_table = lookup_table
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.if_dropout = if_dropout

        if model_name == "lstm_model":
            self.model = rnn.LSTM(input_var,
                                  lookup_table,
                                  n_input,
                                  n_hidden,
                                  n_output)
        if model_name == "lstm_avg_model":
            self.model = rnn.LSTM_avg(input_var,
                                      lookup_table,
                                      n_input,
                                      n_hidden,
                                      n_output)

        if model_name == "cnn_model":
            self.model = cnn.CNN(input_var,
                                 lookup_table,
                                 word_dim,
                                 cnn_n_feature_maps,
                                 cnn_window_sizes,
                                 n_output,
                                 True)

        self.h0 = getattr(self.model, 'h0', T.fvector('h0'))
        self.c0 = getattr(self.model, 'c0', T.fvector('c0'))
        # not used
        self.th = T.fvector('th')
        self.tc = T.fvector('tc')
        self.dt = T.fmatrix('dt')
        self.yt = T.fmatrix('yt')
        self.yt_pred = T.fmatrix('yt_pred')
        self.if_train_var = T.scalar('if_train_var')

        self.relations = T.ivector('relations')
        if model_name != "cnn_model":
            self.level1_hidden = self.model.n_hidden
            self.level2_hidden = 1
            self.mask = self.model.mask
        self.model.build_network()
        self.y = self.model.y
        self.output_layer = layer.OutputLayer(n_output,
                                              self.y,
                                              if_dropout)

        self.y_pred = self.output_layer.y_pred
        self.y_drop_pred = self.output_layer.y_drop_pred
        self.output = self.output_layer.output
        self.loss = self.output_layer.loss
        self.error = self.output_layer.error
        self.params = self.model.params

        # prepare L2_sqr
        self.L2 = 0
        for p in self.params:
            self.L2 += (p**2).sum()
        self.L2 = T.sqrt(self.L2) * 0.01
        return
