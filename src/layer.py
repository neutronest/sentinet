# -*- coding: utf-8 -*-
import theano.tensor as T

class Layer(object):
    """
    """


    def __init__(self):
        self.params = None
        self.mask_value = 0
        return

    def get_params(self):
        return self.params

    def get_inputmask(x):
        """
        return the crossponding Mask Matrix with input x
        """
        return T.any(T.ones_like(x) * (1. - T.eq(x, self.mask_value)), axis=-1)
