# -*- coding: utf-8 -*-

import pdb
import numpy as np
import theano
import theano.tensor as T
import loss
import utils

#theano.config.compute_test_value = 'warn'

"""============== SRNN ============="""
class SRNN(object):
    """ sequence RNN


    """
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden):

        self.input_var = input_var
        self.sens_pos_var = T.imatrix('sens_pos_var')
        self.n_input = n_input
        self.n_hidden = n_hidden

                # weight define
        self.W_input = utils.shared_uniform((n_input, n_hidden),
                                            dtype=theano.config.floatX,
                                            name='W_input')
        self.W_hidden = utils.shared_uniform((n_hidden, n_hidden),
                                             dtype=theano.config.floatX,
                                             name='W_hidden')


        self.b_h = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='bh')

        self.h0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX,
                                     name='h0')

        self.params = [self.W_input,
                       self.W_hidden,
                       self.b_h,
                       self.h0]
        return

    def _recurrent(self, x_t, h_pre):
        """
        """
        h_t = T.nnet.sigmoid(T.dot(x_t, self.W_input) + \
                             T.dot(h_pre, self.W_hidden) + \
                             self.b_h)
        return h_t

    def _get_hidden_state(self, sen_pos):
        words_var = self.input_var[sen_pos[0]:sen_pos[1]]
        h, _ = theano.scan(fn=self._recurrent,
                                sequences=words_var,
                                n_steps=words_var.shape[0],
                                outputs_info=self.h0)
        # return the last hidden as the sentence representation
        return h[-1]

    def build_network(self):

        self.h, _ = theano.scan(fn=self._get_hidden_state,
                                sequences=self.sens_pos_var,
                                outputs_info=None)

        self.hidden_states_var = self.h
        return


"""
================ END SRNN ==========
================  TRNN =============="""

class TRNN(object):
    """
    """
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden,
                 n_output):
        """
        """
        #RNN.__init__(self, input_var, n_input, n_hidden, n_output)
        self.input_var = input_var
        self.relation_pairs = T.imatrix('relation_pairs')
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.TW_input = utils.shared_uniform((n_input, n_hidden),
                                             dtype=theano.config.floatX,
                                             name='TW_input')
        self.TW_hidden = utils.shared_uniform((n_hidden, n_hidden),
                                              dtype=theano.config.floatX,
                                              name='TW_hidden')
        self.TW_output = utils.shared_uniform((n_hidden, n_output),
                                              dtype=theano.config.floatX,
                                              name='TW_output')
        self.tb_h = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='tbh')
        self.tb_y = utils.shared_zeros((n_output,),
                                       dtype=theano.config.floatX,
                                       name='tby')

        self.th = T.vector('th')
        self.params = [self.TW_input,
                       self.TW_hidden,
                       self.TW_output,
                       self.tb_h,
                       self.tb_y]
        return

    def _recurrent(self, relation_pair, hlist_tm1):
        """
        the relation pairs will begin with 0:-1
        -1 means that the root node has no parent node
        so that we must add one element that start with 1:0
        """

        c = relation_pair[0]
        p = relation_pair[1]

        #h_t = T.nnet.sigmoid(T.dot(self.input_var[c], self.TW_input) + \
        #                     T.dot(hlist_tm1[p+1], self.TW_hidden) + \
        #                     self.tb_h)

        h_t = T.nnet.sigmoid(T.dot(self.input_var[c], self.TW_input) + \
                             T.dot(hlist_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden], self.TW_hidden) + \
                             self.tb_h)
        h_next = T.set_subtensor(hlist_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], h_t)
        y = T.dot(h_t, self.TW_output) + self.tb_y
        #hlist_next = T.concatenate([hlist_tm1, h_t.dimshuffle('x', 0)])
        return h_next, y

    def build_network(self):
        [self.h, self.y], _ = theano.scan(fn=self._recurrent,
                                          sequences=self.relation_pairs,
                                          outputs_info=[self.th, None])
        return

"""
==================== END TRNN ==============
==================== BEGIN GRU =============

"""

class SGRU(object):
    """ the sequence GRU"""
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden):
        self.input_var = input_var
        self.sens_pos_var = T.imatrix('sens_pos_var')
        self.n_input = n_input
        self.n_hidden = n_hidden

        # weight define
        self.W_z = utils.shared_uniform((n_input, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='GRU_W_z')
        self.U_z = utils.shared_uniform((n_hidden, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='GRU_U_z')
        self.b_z = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='GRU_b_z')

        self.W_r = utils.shared_uniform((n_input, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='GRU_W_r')
        self.U_r = utils.shared_uniform((n_hidden, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='GRU_U_r')
        self.b_r = utils.shared_zeros((n_hidden,),
                                        dtype=theano.config.floatX,
                                        name='GRU_b_r')

        self.W_h = utils.shared_uniform((n_input, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='GRU_W_h')
        self.U_h = utils.shared_uniform((n_hidden, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='GRU_U_h')

        self.b_h = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='GRU_b_h')

        self.h0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX,
                                     name='GRU_h0')



        self.params = [self.W_z,
                       self.U_z,
                       self.b_z,
                       self.W_r,
                       self.U_r,
                       self.b_r,
                       self.W_h,
                       self.U_h,
                       self.b_h,
                       self.h0
                       ]

        return
    def _recurrent(self, x_t, h_tm1):

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_r) + T.dot(h_tm1, self.U_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_z) + T.dot(h_tm1, self.U_z) + self.b_z)
        h_c = T.tanh(T.dot(x_t, self.W_h) + T.dot((r_t * h_tm1), self.U_h) + self.b_h)
        h = (1-z_t) * h_tm1 + z_t * h_c
        return h

    def _get_h(self, sen_pos):

        words_var = self.input_var[sen_pos[0]:sen_pos[1]]
        h, _ = theano.scan(fn=self._recurrent,
                           sequences=words_var,
                           outputs_info=self.h0)
        return h[-1]

    def build_network(self):

        self.h, _ = theano.scan(fn=self._get_h,
                           sequences=self.sens_pos_var,
                           outputs_info=None)
        return

""" ================== END SGRU ==================

    ================== start TGRU ================
"""
class TGRU(object):
    """
    """
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden,
                 n_output):

        self.input_var = input_var
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.relation_pairs = T.imatrix('relation_pairs')
        self.th = T.vector('th')

        self.W_z = utils.shared_uniform((n_input, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='TGRU_W_z')
        self.U_z = utils.shared_uniform((n_hidden, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='TGRU_U_z')
        self.b_z = utils.shared_zeros((n_hidden,),
                                        dtype=theano.config.floatX,
                                        name='TGRU_b_z')

        self.W_r = utils.shared_uniform((n_input, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='TGRU_W_r')
        self.U_r = utils.shared_uniform((n_hidden, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='TGRU_U_r')
        self.b_r = utils.shared_zeros((n_hidden,),
                                        dtype=theano.config.floatX,
                                        name='TGRU_b_r')

        self.W_h = utils.shared_uniform((n_input, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='TGRU_W_h')
        self.U_h = utils.shared_uniform((n_hidden, n_hidden),
                                        dtype=theano.config.floatX,
                                        name='TGRU_U_h')
        self.b_h = utils.shared_zeros((n_hidden,),
                                        dtype=theano.config.floatX,
                                        name='TGRU_b_h')
        self.TW_output = utils.shared_uniform((n_hidden, n_output),
                                              dtype=theano.config.floatX,
                                              name='TGRU_W_output')
        self.b_y = utils.shared_zeros((n_output,),
                                      dtype=theano.config.floatX,
                                      name='TGRU_b_y')

        self.params = [self.W_z,
                       self.U_z,
                       self.b_z,
                       self.W_r,
                       self.U_r,
                       self.b_r,
                       self.W_h,
                       self.U_h,
                       self.b_h,
                       self.TW_output,
                       self.b_y]
        return


    def _recurrent(self, relation_pair, h_tm1):
        """

        """
        c = relation_pair[0]
        p = relation_pair[1]
        x_t = self.input_var[c]
        h_p = h_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_r) + \
                             T.dot(h_p, self.U_r) + \
                             self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_z) + \
                             T.dot(h_p, self.U_z) + \
                             self.b_z)

        h_c = T.tanh(T.dot(x_t, self.W_h) + \
                     T.dot((r_t*h_p), self.U_h) + \
                     self.b_h)
        h_t = (1 - z_t) * h_p + z_t * h_c
        h_next = T.set_subtensor(h_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], h_t)
        y_t = T.dot(h_t, self.TW_output) + self.b_y
        return h_next, y_t
    def build_network(self):
        [self.h, self.y], _ = theano.scan(fn=self._recurrent,
                                          sequences=self.relation_pairs,
                                          outputs_info=[self.th, None])
        return

"""
==================== END TGRU ========================

==================== START SLSTM =====================

"""

class SLSTM(object):
    """ the sequence LSTM """
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden):
        self.input_var = input_var
        self.sens_pos_var = T.imatrix('sens_pos_var')
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.W_i = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_W_i')
        self.U_i = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_U_i')
        self.b_i = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='SLSTM_b_i')

        self.W_f = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_W_f')
        self.U_f = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_U_f')
        self.b_f = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='SLSTM_b_f')

        self.W_c = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_W_c')
        self.U_c = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_U_c')
        self.b_c = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='SLSTM_b_c')

        self.W_o = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_W_o')
        self.U_o = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='SLSTM_U_o')
        self.b_o = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='SLSTM_b_o')

        self.h0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX,
                                     name='SLSTM_h0')

        self.c0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX,
                                     name='SLSTM_c0')

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.U_c,
                       self.W_o, self.U_o, self.b_o,
                       self.h0, self.c0]
        return

    def _recurrent(self, x_t, h_tm1, c_tm1):
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + \
                             T.dot(h_tm1, self.U_i) + \
                             self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + \
                             T.dot(h_tm1, self.U_f) + \
                             self.b_f)
        # c candiate
        c_c = T.tanh(T.dot(x_t, self.W_c) + \
                     T.dot(h_tm1, self.U_c) + \
                     self.b_c)
        c_t = i_t * c_c + f_t * c_tm1
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + \
                             T.dot(h_tm1, self.U_o) + \
                             self.b_o)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def _get_h(self, sen_pos):
        words_var = self.input_var[sen_pos[0]:sen_pos[1]]
        [h, c], _ = theano.scan(fn=self._recurrent,
                                sequences=words_var,
                                outputs_info=[self.h0, self.c0])
        return h[-1]

    def build_network(self):
        """
        """
        self.h, _ = theano.scan(fn=self._get_h,
                           sequences=self.sens_pos_var,
                           outputs_info=None)
        return

class TLSTM(object):
    """
    Tree LSTM (from top to bottom..)
    """
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden,
                 n_output):
        self.input_var = input_var
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.relation_pairs = T.imatrix('relation_pairs')
        self.th = T.vector('th')
        self.tc = T.vector('tc')

        self.W_i = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_W_i')
        self.U_i = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_U_i')
        self.b_i = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='TLSTM_b_i')

        self.W_f = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_W_f')
        self.U_f = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_U_f')
        self.b_f = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='TLSTM_b_f')

        self.W_c = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_W_c')
        self.U_c = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_U_c')
        self.b_c = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='TLSTM_b_c')

        self.W_o = utils.shared_orthogonal((n_input, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_W_o')
        self.U_o = utils.shared_orthogonal((n_hidden, n_hidden),
                                           scale=1.,
                                           dtype=theano.config.floatX,
                                           name='TLSTM_U_o')
        self.b_o = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='TLSTM_b_o')


        self.TW_output = utils.shared_uniform((n_hidden, n_output),
                                              dtype=theano.config.floatX,
                                              name='TLSTM_W_output')
        self.b_y = utils.shared_zeros((n_output,),
                                      dtype=theano.config.floatX,
                                      name='TLSTM_b_y')

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.U_c,
                       self.W_o, self.U_o, self.b_o,
                       self.TW_output, self.b_y]

        return

    def _recurrent(self, relation_pair, h_tm1, c_tm1):
        """
        """
        c = relation_pair[0]
        p = relation_pair[1]
        x_t = self.input_var[c]
        h_p = h_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]
        c_p = c_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]

        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + \
                             T.dot(h_p, self.U_i) + \
                             self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + \
                             T.dot(h_p, self.U_f) + \
                             self.b_f)
        # c candiate
        c_c = T.tanh(T.dot(x_t, self.W_c) + \
                     T.dot(h_p, self.U_c) + \
                     self.b_c)
        c_t = i_t * c_c + f_t * c_p
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + \
                             T.dot(h_p, self.U_o) + \
                             self.b_o)
        h_t = o_t * T.tanh(c_t)

        h_next = T.set_subtensor(h_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], h_t)
        c_next = T.set_subtensor(c_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], c_t)
        y_t = T.dot(h_t, self.TW_output) + self.b_y
        return h_next, c_next, y_t

    def build_network(self):
        [self.h, self.c, self.y], _ = theano.scan(fn=self._recurrent,
                                                  sequences=self.relation_pairs,
                                                  outputs_info=[self.th, self.tc, None])
        return

class RNN_OneStep(object):
    """ the simple RNN without "scan".
    """

    def __init__(self,
                 input_var,
                 y_pre_var,
                 n_input,
                 n_hidden,
                 n_output,
                 h_tm1):
        """
        the init

        Parameters:
        -----------
        input_var: the input theano variable
            type: theano tensor of dvector

        y_pre_var: the previous y vector
            type: theano tensor of dvector


        n_input: the size of input layer
            type: int

        n_hidden: the size of hidden layer
            type: int

        n_output: the sizs of output layer
            type: int

        h_tm1: the previous hidden theano tensor variable
            type: theano variable of dvector
        """

        self.input_var = input_var
        self.y_pre_var = y_pre_var
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.h_tm1 = h_tm1
        # weight define
        self.W_input = utils.shared_uniform((n_input, n_hidden),
                                     dtype=theano.config.floatX,
                                     name='W_in')
        self.W_hidden = utils.shared_uniform((n_hidden, n_hidden),
                                      dtype=theano.config.floatX,
                                      name='W_hidden')
        self.W_output = utils.shared_uniform((n_hidden, n_output),
                                      dtype=theano.config.floatX,
                                      name='W_output')
        self.W_hy = utils.shared_uniform((n_output, n_hidden),
                                         dtype=theano.config.floatX,
                                         name='W_hy')

        self.b_h = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='bh')
        self.b_y = utils.shared_zeros((n_output,),
                                      dtype=theano.config.floatX,
                                      name='by')

        self.params = [self.W_input,
                       self.W_hidden,
                       self.W_output,
                       self.W_hy,
                       self.b_h,
                       self.b_y]

        return

    def _recurrent(self, x_t, h_pre):
        h_t = T.nnet.sigmoid(T.dot(x_t, self.W_input) \
                             + T.dot(h_pre, self.W_hidden) \
                             + T.dot(self.y_pre_var, self.W_hy) \
                             + self.b_h)
        y_t = T.dot(h_t, self.W_output) + self.b_y
        return h_t, y_t

    def build_network(self):

        self.h, self.y_t = self._recurrent(self.input_var, self.h_tm1)
        self.y_pred = T.nnet.softmax(self.y_t)
        self.output_var = T.argmax(self.y_pred, axis=1)
        self.loss = loss.nll_multiclass
        self.error = loss.mean_classify_error
        return

    # end RNN_OneStep ===================================
class GRU_OneStep(object):
    """
    """
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden,
                 n_output,
                 h_tm1):
        """
        the init

        Parameters:
        -----------
        input_var: the input theano variable
            type: theano tensor of dvector

        n_input: the size of input layer
            type: int

        n_hidden: the size of hidden layer
            type: int

        n_output: the sizs of output layer
            type: int

        h_tm1: the previous hidden theano tensor variable
            type: theano variable of dvector
        """
        self.input_var = input_var
        self.n_input = n_input
        self.n_hideen = n_hidden
        self.n_output = n_output
        self.h_tm1 = h_tm1

        return
