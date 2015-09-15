# -*- coding: utf-8 -*-
import numpy as np
import theano

def wrap_x(x):
    """

    """
    data_x = np.asarray(x, dtype=theano.config.floatX)
    return data_x


def wrap_y(y):
    """
    """
    data_y = np.asarray(y, dtype=np.int32)
    return data_y
