# -*- coding: utf-8 -*-

import pdb
import numpy as np
import theano
import theano.tensor as T


from .utils import sharedX, uniform, shared_zero, shared_ones

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
                 activation="sigmoid",
                 output_type="softmax",
                 if_dropout=True):
        """
        the rnn init function
        Parameters:
        -----------


        """

        self.input_data = input_data
        self.activation = activation
        self.output_type = output_type
        self.n_hidden = n_hidden

        # recurrent weights as a shared variable
        W_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                              low=-.01, high=.01),
                                              dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name='W')
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

        self.params = [self.W, self.W_in, self.W_out, self.h0,
                       self.bh, self.by]

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        def step(x_t, h_previous):
            """

            """

            h_t = T.nnet.sigmoid(T.dot(x_t, self.W_in) + T.dot(h_previous, self.W) + self.bh)

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


        if self.output_type == "softmax":
            self.p_y_given_x_var = T.nnet.softmax(self.y_pred_var)
            self.output_var = T.argmax(self.p_y_given_x_var, axis=1)
            self.loss = self.nll_multiclass # point-free oh~yeah

        ## ==== end function ===

    def nll_multiclass(self, y):
        return -T.mean(T.log(self.p_y_given_x_var)[T.arange(y.shape[0]), y])

    def error(self, label):
        """
        Parameter:
        label: the real category of data
               type: list(int)
        """
        return T.mean(T.neq(self.output_var, label))



class RNN_LSTM(RNN):
    """
    the recurrent neural network with LSTM
    """
    def W_init(n_row, n_col, w_name):
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
    def get_fans(shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out

    def glorot_uniform(shape):
        fan_in, fan_out = get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return uniform(shape, s)

    def orthogonal(shape, scale=1.1):
        ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
        '''
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
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

        super(LSTM, self).__init__()
        # weights of LSTM
        self.W_i = self.glorot_uniform((n_input, n_hidden))
        self.U_i = self.orthogonal((n_hidden, n_hidden))
        self.b_i = shared_zero((n_hidden,))

        self.W_f = self.glorot_uniform((n_input, n_hidden))
        self.U_f = self.glorot_uniform((n_hidden, n_hidden))
        self.b_f = shared_ones((n_hidden,))

        self.W_c = self.glorot_uniform((n_input, n_hidden))
        self.U_c = self.orthogonal((n_hidden, n_hidden))
        self.b_c = shared_zero((n_hidden,))

        self.W_o = self.glorot_uniform((n_input, n_hidden))
        self.U_o = self.orthogonal((n_hidden, n_hidden))
        self.b_o = shared_zero((n_hidden,))
