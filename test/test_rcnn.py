## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys

sys.path.append("../src")
import rnn
import cnn
import rcnn
import test_cnn
import test_rnn
import test_init
def test_rcnn():
    """
    """
    rng=np.random.RandomState(54321)
    x_var = T.dtensor3('x_var')
    word_dim = 300
    cnn_n_feature_maps = 300
    cnn_window_sizes = (2,3,4)
    rnn_n_hidden = 300
    rnn_n_out = 3
    rcnn_model = rcnn.RCNN(rng=rng,
                           input_data=x_var,
                           dim=word_dim,
                           n_feature_maps=cnn_n_feature_maps,
                           window_sizes=cnn_window_sizes,
                           n_hidden=rnn_n_hidden,
                           n_out=rnn_n_out)



    (_, y_true_var, y_pred_var, _, seq_word_x, y_true, y_pred) = test_init.generate_test_data()
    lr_var = T.scalar('lr_var')
    cost_var = rcnn_model.loss(y_true_var, rcnn_model.y)
    
    optimizer_updates = {}
    gparams = [T.grad(cost_var, param_var) for param_var in rcnn_model.params]
    optimizer_updates = [(param, param - gparam * lr_var) \
        for param, gparam in zip(rcnn_model.params, gparams)]

    compute_cost_fn = theano.function(inputs=[rcnn_model.input_var, y_true_var],
                                      outputs=cost_var)
    compute_train_fn = theano.function(inputs=[rcnn_model.input_var, y_true_var, lr_var],
                                       outputs=cost_var,
                                       updates=optimizer_updates)
    print "[Begin to test rcnn cost function]"
    cost = compute_cost_fn(seq_word_x, y_true)
    print "the rcnn compute cost is %f" %(cost)

    print "[Begin to test rcnn train function]"
    compute_train_fn(seq_word_x, y_true, 0.99)
    print "train success!"
    
    print "[Test rcnn OK!]"
    print "====="
    return

if __name__ == "__main__":
    test_rcnn()
