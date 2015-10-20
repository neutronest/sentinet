# -*- coding: utf-8 -*-

import pdb
import numpy as np
import theano
import theano.tensor as T
import loss
from utils import sharedX, uniform, shared_zeros, shared_ones

class RNN(object):
    """
    the basic recurrent neural network
    """

    def __init__(self,
                 rng,
                 input_data,
                 n_input,
                 n_hidden,
                 n_output,
                 activation=T.nnet.sigmoid,
                 output_type="softmax",
                 if_dropout=True):
        """
        the rnn init function
        Parameters:
        -----------


        """

        self.input_data = input_data
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # recurrent weights as a shared variable
        W_h = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                              low=-.01, high=.01),
                                              dtype=theano.config.floatX)
        self.W_h = theano.shared(value=W_h, name='W_h')
        # input to hidden layer weights
        W_in_init = np.asarray(np.random.uniform(size=(n_input, n_hidden),
                                                 low=-.01, high=.01),
                                                 dtype=theano.config.floatX)
        self.W_in = theano.shared(value=W_in_init, name='W_in')

        # hidden to output layer weights
        W_out_init = np.asarray(np.random.uniform(size=(n_hidden, n_output),
                                                  low=-.01, high=.01),
                                                  dtype=theano.config.floatX)
        self.W_out = theano.shared(value=W_out_init, name='W_out')

        h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init, name='h0')

        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = np.zeros((n_output,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init, name='by')

        self.params = [self.W_h, self.W_in, self.W_out, self.h0,
                       self.bh, self.by]
                       
        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        def step(x_t, h_previous):
            """

            """

            h_t = T.nnet.sigmoid(T.dot(x_t, self.W_in) + T.dot(h_previous, self.W_h) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by

            # dropout layer
            """
            if if_dropout:
                srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
                mask = srng.binomial(n=1,
                                     p=0.5,
                                     size=y_t.shape)
                y_t = y_t * T.cast(mask, theano.config.floatX)
            """
            return h_t, y_t

        [self.h_var, self.y_pred_var], _ = theano.scan(step,
                                                       sequences=self.input_data,
                                                       outputs_info=[self.h0, None])

        self.p_y_given_x_var = T.nnet.softmax(self.y_pred_var)
        self.output_var = T.argmax(self.p_y_given_x_var, axis=1)
        self.loss = loss.mean_binary_crossentropy
        self.error = loss.mean_classify_error
        ## ==== end function ===
        return

class RNN_LSTM(object):
    """
    the recurrent neural network with LSTM
    """
    def W_init(self, n_row, n_col, w_name):
        """
        """
        W = np.asarray(np.random.uniform(size=(n_row, n_col),
                                         low=-.01,
                                         high=.01),
                       dtype=theano.config.floatX)
        return theano.shared(value=W, name=w_name)

    """
    copying from keras
    not use keras right now.
    """
    def get_fans(self, shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out

    def glorot_uniform(self, shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return uniform(shape, s)

    def orthogonal(self, shape, scale=1.1):
        ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
        '''
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshapeo(shape)
        return sharedX(scale * q[:shape[0], :shape[1]])

    def __init__(self,
                 rng,
                 input_data,
                 n_input,
                 n_hidden,
                 n_output,
                 activation="sigmoid",
                 output_type="softmax",
                 if_dropout=True):


        # weights of LSTM
        self.W_i = self.glorot_uniform((n_input, n_hidden))
        self.U_i = self.orthogonal((n_hidden, n_hidden))
        self.b_i = shared_zeros((n_hidden,))

        self.W_f = self.glorot_uniform((n_input, n_hidden))
        self.U_f = self.glorot_uniform((n_hidden, n_hidden))
        self.b_f = shared_ones((n_hidden,))

        self.W_c = self.glorot_uniform((n_input, n_hidden))
        self.U_c = self.orthogonal((n_hidden, n_hidden))
        self.b_c = shared_zeros((n_hidden,))

        self.W_o = self.glorot_uniform((n_input, n_hidden))
        self.U_o = self.orthogonal((n_hidden, n_hidden))
        self.b_o = shared_zeros((n_hidden,))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_c, self.U_c, self.b_c,
            self.W_o, self.U_o, self.b_o
        ]

        # LSTM Cell step
        def _step(self, \
                  xi_t, xf_t, xo_t, xc_t, \
                  mask_tm1, h_tm1, c_tm1,\
                  u_i, u_f, u_o, u_c):
            h_mask_tm1 = mask_tm1 * h_tm1
            c_mask_tm1 = mask_tm1 * c_tm1

            i_t = T.nnet.hard_sigmoid(xi_t + T.dot(h_mask_tm1, u_i))
            f_t = T.nnet.hard_sigmoid(xf_t + T.dot(h_mask_tm1, u_f))
            c_t = f_t * c_tm1 + i_t * T.tanh(xc_t + T.dot(h_mask_tm1, u_c))
            o_t = T.nnet.hard_sigmoid(xo_t + T.dot(h_mask_tm1, u_o))
            h_t = o_t * T.tanh(c_t)
            return h_t, c_t

        return

    def adadelta_optimizer(self, params, loss):

        def _clip_norm(g, c, norm):
            if c > 0:
                g = T.swtich(T.ge(n, c), g * c / n, g)
            return g


        def _get_gradients(params, loss):
            """
            """
            grads = T.grad(loss, params)
            norm = T.sqrt(sum([g ** 2 for g in grads]))
