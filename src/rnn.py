# -*- coding: utf-8 -*-

import pdb
import numpy as np
import theano
import theano.tensor as T
import loss
import utils
import config
#theano.config.compute_test_value = 'warn'

"""============== SRNN ============="""
class SRNN(object):
    """ sequence RNN


    """
    def __init__(self,
                 input_var,
                 lookup_table,
                 n_input,
                 n_hidden):
        """
        apply a dtensor3 data
        """
        self.input_var = input_var
        self.lookup_table = utils.sharedX(lookup_table,
                                          dtype=theano.config.floatX,
                                          name='W_emb')
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.mask = T.fmatrix('SRNN_mask')
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
        self.h0 = T.fmatrix('h0')

        self.params = [self.lookup_table,
                       self.W_input,
                       self.W_hidden,
                       self.b_h]
        return

    def _recurrent(self, x_t, m_t, h_pre):
        """
        """
        #x_emb = self.lookup_table[x_t]
        h_ct = T.nnet.sigmoid(T.dot(x_t, self.W_input) + \
                             T.dot(h_pre, self.W_hidden) + \
                             self.b_h)
        h_t = m_t[:, None] * h_ct + (1 - m_t)[:, None] * h_pre

        return h_t

    def build_network(self):
        x_emb = self.lookup_table[self.input_var.flatten()].reshape([self.input_var.shape[0],
                                                                     self.input_var.shape[1],
                                                                     config.options['word_dim']])
        self.h_history, _ = theano.scan(fn=self._recurrent,
                                sequences=[x_emb, self.mask],
                                outputs_info=[self.h0])
        self.h = self.h_history[-1]
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
        self.relations = T.ivector('relations')
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

        self.th = T.fvector('th')
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

        c = relations[0]
        p = relations[1]

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
                 lookup_table,
                 n_input,
                 n_hidden,
                 sigle_mode=False,
                 n_output=None):
        self.input_var = input_var
        self.lookup_table = utils.sharedX(lookup_table,
                                          dtype=theano.config.floatX,
                                          name='W_emb')
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.mask = T.fmatrix('SGRU_mask')
        self.h0 = T.fmatrix('SGRU_h0')
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


        self.params = [self.W_z,
                       self.U_z,
                       self.b_z,
                       self.W_r,
                       self.U_r,
                       self.b_r,
                       self.W_h,
                       self.U_h,
                       self.b_h,
                       self.lookup_table
                       ]

        return
    def _recurrent(self, x_t, m_t, h_tm1):
        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_r) + T.dot(h_tm1, self.U_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_z) + T.dot(h_tm1, self.U_z) + self.b_z)
        h_c = T.tanh(T.dot(x_t, self.W_h) + T.dot((r_t * h_tm1), self.U_h) + self.b_h)
        h_m = (1-z_t) * h_tm1 + z_t * h_c

        h = m_t[:, None] * h_m + (1-m_t)[:, None] * h_tm1
        return h

    def build_network(self):

        x_emb = self.lookup_table[self.input_var.flatten()].reshape([self.input_var.shape[0],
                                                                     self.input_var.shape[1],
                                                                     config.options['word_dim']])

        self.h_history, _ = theano.scan(fn=self._recurrent,
                                sequences=[x_emb,
                                           self.mask],
                                outputs_info=[self.h0])
        self.h = self.h_history[-1]
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
        self.relations = T.ivector('relations')
        self.th = T.fvector('th')

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


    def _recurrent(self, idx, h_tm1, r):
        """

        """
        c = idx
        p = r[idx]
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
                                          sequences=T.arange(self.relations.shape[0]),
                                          non_sequences=self.relations,
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
                 lookup_table,
                 n_input,
                 n_hidden):
        self.input_var = input_var
        self.lookup_table =  utils.sharedX(lookup_table,
                                           dtype=theano.config.floatX,
                                           name='W_emb')
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.mask = T.fmatrix('SLSTM_mask')
        self.h0 = T.fmatrix('SLSTM_h0')
        self.c0 = T.fmatrix('SLSTM_c0')

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
        #TODO: TRICK
        #self.b_f = utils.shared_zeros((n_hidden,),
        #                              dtype=theano.config.floatX,
        #                              name='SLSTM_b_f')
        self.b_f = utils.sharedX(np.ones((n_hidden),
                                         dtype=theano.config.floatX),
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

        """
        self.h0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX,
                                     name='SLSTM_h0')

        self.c0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX,
                                     name='SLSTM_c0')
        """
        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.U_c,
                       self.W_o, self.U_o, self.b_o,
                       self.lookup_table]
        return

    def _recurrent(self, x_t, m_t, h_tm1, c_tm1):

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
        c_mt = i_t * c_c + f_t * c_tm1
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + \
                             T.dot(h_tm1, self.U_o) + \
                             self.b_o)
        h_mt = o_t * T.tanh(c_mt)

        h_t = m_t[:, None] * h_mt + (1 - m_t)[:, None] * h_tm1
        c_t = m_t[:, None] * c_mt + (1 - m_t)[:, None] * c_tm1

        return h_t, c_t

    def build_network(self):
        """
        """

        x_emb = self.lookup_table[self.input_var.flatten()].reshape([self.input_var.shape[0],
                                                                     self.input_var.shape[1],
                                                                     config.options['word_dim']])
        [self.h_history, self.c], _ = theano.scan(fn=self._recurrent,
                                                  sequences=[x_emb,
                                                             self.mask],
                                                  outputs_info=[self.h0,
                                                                self.c0])
        self.h = self.h_history[-1]
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
        self.relations = T.ivector('relations')
        self.th = T.fvector('th')
        self.tc = T.fvector('tc')

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

    def _recurrent(self, idx, h_tm1, c_tm1, r):
        """
        """
        c = idx
        p = r[idx]
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
                                                  sequences=T.arange(self.relations.shape[0]),
                                                  non_sequences=self.relations,
                                                  outputs_info=[self.th, self.tc, None])
        return



class SLSTM_avg(SLSTM):
    """
    using avarage hidden of LSTM
    """
    def __init__(self,
                 input_var,
                 lookup_table,
                 n_input,
                 n_hidden):
        SLSTM.__init__(self,
                       input_var,
                       lookup_table,
                       n_input,
                       n_hidden)
        return

    def build_network(self):
        x_emb = self.lookup_table[self.input_var.flatten()].reshape([self.input_var.shape[0],
                                                                     self.input_var.shape[1],
                                                                     config.options['word_dim']])
        [self.h_history, self.c], _ = theano.scan(fn=self._recurrent,
                                                  sequences=[x_emb,
                                                             self.mask],
                                                  outputs_info=[self.h0,
                                                                self.c0])
        self.h = T.mean(self.h_history, axis=0)
        return


class TLSTM_s(TLSTM):
    """
    a strong version of TLSTM
    """
    def __init__(self,
                input_var,
                n_input,
                n_hidden,
                n_output):
        TLSTM.__init__(self,
                      input_var,
                      n_input,
                      n_hidden,
                      n_output)

        self.P_i = utils.shared_orthogonal((n_hidden, n_hidden),
                                           dtype=theano.config.floatX,
                                           name="TLSTM_P_i")
        self.P_f = utils.shared_orthogonal((n_hidden, n_hidden),
                                           dtype=theano.config.floatX,
                                           name="TLSTM_P_f")
        self.P_o = utils.shared_orthogonal((n_hidden, n_hidden),
                                           dtype=theano.config.floatX,
                                           name="TLSTM_P_o")
        self.params += [self.P_i, self.P_f, self.P_o]

        return

    def _recurrent(self, idx, h_tm1, c_tm1, r):
        """
        """
        c = idx
        p = r[idx]
        x_t = self.input_var[c]
        h_p = h_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]
        c_p = c_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]

        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + \
                             T.dot(h_p, self.U_i) + \
                             T.dot(c_p, self.P_i) + \
                             self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + \
                             T.dot(h_p, self.U_f) + \
                             T.dot(c_p, self.P_f) + \
                             self.b_f)
        # c candiate
        c_c = T.tanh(T.dot(x_t, self.W_c) + \
                     T.dot(h_p, self.U_c) + \
                     self.b_c)
        c_t = i_t * c_c + f_t * c_p
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + \
                             T.dot(h_p, self.U_o) + \
                             T.dot(c_t, self.P_o) + \
                             self.b_o)
        h_t = o_t * T.tanh(c_t)

        h_next = T.set_subtensor(h_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], h_t)
        c_next = T.set_subtensor(c_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], c_t)
        y_t = T.dot(h_t, self.TW_output) + self.b_y
        return h_next, c_next, y_t


class TLSTM_f(TLSTM):

    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden,
                 n_output):
        TLSTM.__init__(self,
                       input_var,
                       n_input,
                       n_hidden,
                       n_output)

        self.D_i = utils.shared_orthogonal((config.options['dfeature_len'], n_hidden),
                                             dtype=theano.config.floatX,
                                             name='TLSTMf_D_i')
        self.D_f = utils.shared_orthogonal((config.options['dfeature_len'], n_hidden),
                                             dtype=theano.config.floatX,
                                             name='TLSTMf_D_f')
        self.D_o = utils.shared_orthogonal((config.options['dfeature_len'], n_hidden),
                                             dtype=theano.config.floatX,
                                             name='TLSTMf_D_o')
        self.D_c = utils.shared_orthogonal((config.options['dfeature_len'], n_hidden),
                                             dtype=theano.config.floatX,
                                             name='TLSTMf_D_c')

        self.dt = T.fmatrix('dt')
        self.yt = T.fvector('yt')
        self.yt_pred = T.fvector('yt_pred')
        self.if_train_var = T.scalar('if_train')
        self.params += [self.D_i, self.D_f, self.D_o, self.D_c]
        return

    def _recurrent(self, idx, h_tm1, c_tm1, yt_tm1, r, if_train):
        """
        """
        c = idx
        p = r[idx]
        x_t = self.input_var[c]
        h_p = h_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]
        c_p = c_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]

        d_p = T.switch(if_train,
                       T.concatenate(self.dt[c], self.yt[p+1]),
                       T.concatenate(self.dt[c], self.yt_pred[p+1]))

        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + \
                             T.dot(h_p, self.U_i) + \
                             T.dot(d_p, self.D_i) + \
                             self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + \
                             T.dot(h_p, self.U_f) + \
                             T.dot(d_p, self.D_f) + \
                             self.b_f)
        # c candiate
        c_c = T.tanh(T.dot(x_t, self.W_c) + \
                     T.dot(h_p, self.U_c) + \
                     T.dot(d_p, self.D_c) + \
                     self.b_c)
        c_t = i_t * c_c + f_t * c_p
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + \
                             T.dot(h_p, self.U_o) + \
#                             T.dot(c_t, self.P_o) + \
                             T.dot(d_p, self.D_o) + \
                             self.b_o)
        h_t = o_t * T.tanh(c_t)
        y_t = T.dot(h_t, self.TW_output) + self.b_y
        h_next = T.set_subtensor(h_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], h_t)
        c_next = T.set_subtensor(c_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], c_t)
        self.yt_pred[c+1] = T.argmax(T.nnet.softmax(y_t))
        return h_next, c_next, y_t

    def build_network(self):
        [self.h, self.c, self.y], _ = theano.scan(fn=self._recurrent,
                                                  sequences=T.arange(self.relations.shape[0]),
                                                  non_sequences=[self.relations, self.if_train_var],
                                                  outputs_info=[self.th, self.tc, None])
        return


class TLSTM_fc(TLSTM):
    """
    TLSTM with feature cell.
    more strong version than TLSTM-f
    """
    def __init__(self,
                 input_var,
                 n_input,
                 n_hidden,
                 n_output):
        TLSTM.__init__(self,
                       input_var,
                       n_input,
                       n_hidden,
                       n_output)
        self.dt = T.fmatrix('d_t')
        self.yt = T.fvector('yt')
        self.yt_pred = T.fvector('yt_pred')
        self.if_train_var = T.scalar('if_train')
        # addtional params
        self.W_v = utils.shared_orthogonal((n_input, n_hidden),
                                           dtype=theano.config.floatX,
                                           name="TLSTM_W_v")
        self.U_v = utils.shared_orthogonal((n_hidden, n_hidden),
                                           dtype=theano.config.floatX,
                                           name="TLSTM_U_v")
        self.D_v = utils.shared_orthogonal((config.options['dfeature_len'],
                                            n_hidden),
                                           dtype=theano.config.floatX,
                                           name="TLSTM_D_v")
        self.b_v = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='TLSTM_b_v')

        self.W_dc = utils.shared_orthogonal((config.options['dfeature_len'],
                                             n_hidden),
                                            dtype=theano.config.floatX)
        self.params.append(self.W_v, self.U_v, self.D_v, self.b_v, self.W_dc)
        return

    def _recurrent(self, idx, h_tm1, c_tm1, yt_tm1, r, if_train):

        c = idx
        p = r[idx]
        x_t = self.input_var[c]
        h_p = h_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]
        c_p = c_tm1[(p+1)*self.n_hidden:(p+2)*self.n_hidden]

        d_p = T.switch(if_train,
                       T.concatenate(self.dt[c], self.yt[p+1]),
                       T.concatenate(self.dt[c], self.yt_pred[p+1]))
        d_tm1 = T.swtich(if_train,
                         T.concatenate(self.dt[p], self.yt[r[p]+1]),
                         T.concatenate(self.dt[p], self.yt_pred[r[p]+1]))

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

        v_t = T.nnet.sigmoid(T.dot(x_t, self.W_v) + \
                             T.dot(h_p, self.U_v) + \
                             self.b_v)
        dc_t = (1 - v_t) * d_p + v_t * d_tm1
        c_t = i_t * c_c + f_t * c_p + T.tanh(T.dot(dc_t, self.W_dc))
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + \
                             T.dot(h_p, self.U_o) + \
                             self.b_o)
        h_t = o_t * T.tanh(c_t)
        y_t = T.dot(h_t, self.TW_output) + self.b_y
        h_next = T.set_subtensor(h_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], h_t)
        c_next = T.set_subtensor(c_tm1[(c+1)*self.n_hidden:(c+2)*self.n_hidden], c_t)
        self.yt_pred[c+1] = T.argmax(T.nnet.softmax(y_t))
        return

    def build_network(self):
        [self.h, self.c, self.y], _ = theano.scan(fn=self._recurrent,
                                                  sequences=T.arange(self.relations.shape[0]),
                                                  non_sequences=[self.relations, self.if_train_var],
                                                  outputs_info=[self.th, self.tc, None])
        return



class LSTM(SLSTM):
    def __init__(self,
                 input_var,
                 lookup_table,
                 n_input,
                 n_hidden,
                 n_output):
        self.output = n_output
        SLSTM.__init__(self,
                       input_var,
                       lookup_table,
                       n_input,
                       n_hidden)
        self.W_output = utils.shared_uniform((n_hidden, n_output),
                                             dtype=theano.config.floatX,
                                             name="LSTM_output")
        self.b_y = utils.shared_zeros((n_output),
                                     dtype=theano.config.floatX,
                                     name="LSTM_by")
        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.U_c,
                       self.W_o, self.U_o, self.b_o,
                       self.W_output, self.b_y,
                       self.lookup_table]
        return

    def build_network(self):
        """
        """

        x_emb = self.lookup_table[self.input_var.flatten()].reshape([self.input_var.shape[0],
                                                                     self.input_var.shape[1],
                                                                     config.options['word_dim']])
        [self.h_history, self.c], _ = theano.scan(fn=self._recurrent,
                                                  sequences=[x_emb,
                                                             self.mask],
                                                  outputs_info=[self.h0,
                                                                self.c0])
        self.h = self.h_history[-1]
        self.y = T.dot(self.h, self.W_output) + self.b_y
        return

class LSTM_avg(LSTM):
    """
    single lstm with avg hidden layer
    """
    def __init__(self,
                 input_var,
                 lookup_table,
                 n_input,
                 n_hidden,
                 n_output):
        LSTM.__init__(self,
                      input_var,
                      lookup_table,
                      n_input,
                      n_hidden,
                      n_output)

        return

    def build_network(self):
        """
        """

        x_emb = self.lookup_table[self.input_var.flatten()].reshape([self.input_var.shape[0],
                                                                     self.input_var.shape[1],
                                                                     config.options['word_dim']])
        [self.h_history, self.c], _ = theano.scan(fn=self._recurrent,
                                                  sequences=[x_emb,
                                                             self.mask],
                                                  outputs_info=[self.h0,
                                                                self.c0])
        self.h = T.mean(self.h_history,axis=0)
        self.y = T.dot(self.h, self.W_output) + self.b_y
        return
