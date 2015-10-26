# -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys

sys.path.append("../src")
import models
import utils
import optimizer
from collections import OrderedDict

def test_sgd():

    # param for model prepare
    input_var = T.dmatrix('input_var')
    y_var = T.ivector('y_var')
    word_dim = 300
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

    sgd = optimizer.SGD() # use default params

    print "gparam define"
    gparams_var = [T.grad(cost_var, param) for param in rcnn_onestep_model.params]
    print "param update function and train function define"
    #pdb.set_trace()
    optimize_updates = {}
    for (param_var, gparam_var) in zip(rcnn_onestep_model.params, gparams_var):
        optimize_updates[param_var] = sgd.param_update(param_var, gparam_var)

    model_train_fn = theano.function(inputs=[],
                                     outputs=None,
                                     updates=optimize_updates)


    print "[Test SGD Optimizer OK!]"
    return


if __name__ == "__main__":
    test_sgd()
