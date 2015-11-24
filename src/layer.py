# -*- coding: utf-8 -*-
import numpy as np
import theano, theano.tensor as T
import loss
import utils
class Layer(object):
    """
    """


    def __init__(self):
        self.params = None
        self.mask_value = 0
        return

    def get_params(self):
        return self.params

    def get_inputmask(self, x):
        """
        return the crossponding Mask Matrix with input x
        """
        return T.any(T.ones_like(x) * (1. - T.eq(x, self.mask_value)), axis=-1)

class OutputLayer(object):
    """
    the output layer
    """
    def __init__(self,
                 n_output,
                 y,
                 if_dropout="dropout"):
        self.n_output = n_output
        self.y = y
        self.if_dropout = if_dropout

        if self.if_dropout == "dropout":
            rng = np.random.RandomState(1234)
            self.y_drop = self.dropout_layer(rng, self.y, 0.5)
            self.y_drop_pred = T.nnet.softmax(self.y_drop)
            self.y = self.y * 0.5
            self.y_pred = T.nnet.softmax(self.y)
        else:
            self.y_pred = T.nnet.softmax(self.y)
            self.y_drop_pred = self.y_pred

        self.output = T.argmax(self.y_pred, axis=1)
        self.loss = loss.binary_crossentropy
        self.error = loss.mean_classify_error
        return

    def dropout_layer(self, rng, layer, p):
        srng = T.shared_randomstreams.RandomStreams(
            rng.randint(999999))
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        output = layer * T.cast(mask, theano.config.floatX)
        return output
