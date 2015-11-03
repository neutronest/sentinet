# -*- coding: utf-8 -*-

import pdb
import numpy as np
import theano
import theano.tensor as T
import loss
import utils




class SRNN(object):
    """ sequence RNN


    """
    def __init__(self,
                 input_var,
                 sens_pos,
                 n_input,
                 n_hidden,
                 n_output):

        self.input_var = input_var
        self.sens_pos = sens_pos
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

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

        self.b_h = utils.shared_zeros((n_hidden,),
                                      dtype=theano.config.floatX,
                                      name='bh')
        self.b_y = utils.shared_zeros((n_output,),
                                      dtype=theano.config.floatX,
                                      name='by')

        self.h0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX)

        self.params = [self.W_input,
                       self.W_hidden,
                       self.W_output,
                       self.b_h,
                       self.b_y,
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
        words_var = self.input_var.take(T.arange(sen_pos[0], sen_pos[1]))

        h, inner_updates = theano.scan(fn=self._recurrent,
                                       sequences=words_var,
                                       n_steps=words_var.shape[0],
                                       outputs_info=self.h0)
        # return the last hidden as the sentence representation
        return h[-1]

    def build_network(self):
        self.h, outer_updates = theano.scan(fn=self._get_hidden_state,
                                            sequences=self.sens_pos,
                                            outputs_info=None)

        self.hidden_states = self.h
        self.y_pred_var = T.nnet.softmax(self.y)
        self.output = T.argmax(self.y_pred_var, axis=1)
        self.loss = loss.nll_multiclass
        self.error = loss.mean_classify_error
        return


class TRNN(object):
    """
    """
    def __init__(self,
                 input_var_list,
                 relation_pairs,
                 n_input,
                 n_hidden,
                 n_output):
        """
        """
        #RNN.__init__(self, input_var, n_input, n_hidden, n_output)
        self.input_var_list = input_var_list
        self.relation_pairs = relation_pairs
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.TW_input = utils.shared_uniform((n_input, n_output),
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

        self.th0 = utils.shared_zeros((n_hidden,),
                                     dtype=theano.config.floatX)

        self.params = [self.TW_input,
                       self.TW_hidden,
                       self.TW_output,
                       self.tb_h,
                       self.tb_y,
                       self.th0]
        return

    def _recurrent(self, relation_pair):
        x_t = self.input_var_list[relation_pair[0]]
        h_pre = self.h_state_list[relation_pair[1]+1]

        self.h_state_list[relation_pair[0]+1] = T.nnet.sigmoid(T.dot(x_t, self.TW_input),
                                                               T.dot(h_pre, self.TW_hidden),
                                                               self.tb_h)
        y_t = T.dot(self.h_state_list[relation_pair[0]+1], self.TW_output) + self.tb_y
        return y_t

    def build_network(self):
        self.h_state_list = [utils.shared_zeros((self.n_hidden,), dtype=theano.config.floatX)] * (len(self.input_var_list)+1)

        y, _ = theano.scan(fn=self._recurrent,
                                       sequences=self.relation_pairs,
                                       outputs_info=None)
        self.y_pred = T.nnet.softmax(y)
        self.output = T.argmax(self.y_pred, axis=1)
        self.loss = loss.nll_multiclass
        self.error = loss.mean_classify_error
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
