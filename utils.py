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

def expand_y(label, n_class, dtype=np.int32):
    """
    """
    label_vector = np.zeros((n_class,))
    label_vector[label] = 1
    return np.asarray(label_vector, dtype=dtype)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def uniform(shape, scale=0.05):
        return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))

def shared_zero(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape, dtype=dtype), name=name)

def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape, dtype=dtype), name=name)
