# -*- coding: utf-8 -*-

import theano, theano.tensor as T
import numpy as np

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7

def binary_crossentropy(y_true, y_pred):
    """
    calculate the binary cross entropy

    Parameters:
    -----------

    y_true: the prob distribution of actual output
    type: theano.variable dtype=int32/int64

    y_pred: the prob distribution of predict output
    type: theano.variable dtype=float32/64

    """

    # tricks
    # limit the y_pred at the [epsilon, 1-epsilon]

    # y_pred = T.clip(y_pred, epsilon, 1-epsilon)
    #return T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return T.mean(T.nnet.binary_crossentropy(y_pred, y_true))

def nll_multiclass(y_true, y_pred):
    """
    """

    y_pred /= y_pred.sum(axis=-1,keepdims=True)
    #return -T.mean(T.log(y_pred)[T.arange(y_true.shape[0]), y_true])
    return T.mean(T.nnet.categorical_crossentropy(y_pred, y_true))

def mean_classify_error(label_true, label_pred):
    """
    calculate the predict error between test data and predict result

    Parameters:
    -----------

    label_true: the actual label of data
    type: theano.variable dtype=np.int32

    label_pred: the predict label of data
    type: theano.variable dtype=np.int32
    """
    return T.neq(label_true, label_pred)
