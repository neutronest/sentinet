# -*- coding: utf-8 -*-

import pdb
import numpy as np
import theano
import theano.tensor as T

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
        pdb.set_trace()
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
