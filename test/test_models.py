# -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys

sys.path.append("../src")
import rnn
import cnn
import models
import utils
def test_rcnn_onestep():
    """
    """
    # prepare dataset

    sen_len = 10
    word_dim = 300
    # prepare data
    sen = []
    for i in xrange(sen_len):
        word_emb = utils.ndarray_uniform((word_dim,), dtype=theano.config.floatX)
        sen.append(word_emb)

    sen_x = np.asarray(sen, dtype=theano.config.floatX)
    h_0 = utils.ndarray_uniform((300,), 0.05)
    y_true = np.asarray([0, 1, 0], dtype=np.int32)

    # param
    input_var = T.dmatrix('input_var')
    y_var = T.ivector('y_var')
    cnn_feature_maps = 300
    cnn_window_sizes = (2,3,4)
    rnn_hidden = 300
    rnn_output = 3
    h_tm1 = T.dvector('h_tm1')

    rcnn_onestep_model = models.RCNN_OneStep(input_var,
                                       word_dim,
                                       cnn_feature_maps,
                                       cnn_window_sizes,
                                       rnn_hidden,
                                       rnn_output,
                                       h_tm1)
    cost_var = rcnn_onestep_model.loss(y_var, rcnn_onestep_model.y_pred)
    compute_cost_fn = theano.function(inputs=[input_var, y_var, h_tm1],
                                      outputs=[cost_var, rcnn_onestep_model.h])

    [cost, h_current] = compute_cost_fn(sen_x, y_true, h_0)
    print "the cost of rcnn_model_onestep is %f"%(cost)
    print "the current h of rcnn_model_onestep is :"
    print h_current
    print "[Test rcnn_model_onestep OK!]"
    print "====="
    return

if __name__ == "__main__":
    test_rcnn_onestep()
