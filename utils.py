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

def expand_y(y_seq, n_class, dtype=np.int32):
    """
    Parameters:
    -----------
    y_seq: a sequences of labal
       type: list of int

    n_class: the num of classes
    """
    y_matrix = []
    for y in y_seq:
        y_vec = [0] * n_class
        y_vec[y] = 1
        y_matrix.append(y_vec)

    return np.asarray(y_matrix, dtype=dtype)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def uniform(shape, scale=0.05):
        return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))

def shared_zero(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape, dtype=dtype), name=name)

def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape, dtype=dtype), name=name)
