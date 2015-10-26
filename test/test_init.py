## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys
sys.path.append("../src")
import utils

def generate_test_data_onestep():
    """
    only generate data with batch=1
    """

    sen_emb = utils.uniform((10,1), 0.1, theano.config.floatX)
    # TODO!
    return

def generate_test_data():
    # parameters prepare
    seq_len = 5
    sen_len = 10
    n_input = 900
    n_hidden = 300
    n_output = 3
    word_dim = 300
    # sequence data prepare
    seq = []
    for i in xrange(seq_len):
        sen_emb = np.asarray(np.random.uniform(size=(n_input,),
                             low=-.01, high=.01),
                             dtype=theano.config.floatX)
        seq.append(sen_emb)
    seq_x = np.asarray(seq, dtype=theano.config.floatX)


    seq_word = []
    for sen_iter in xrange(seq_len):
        sen = []
        for i in xrange(sen_len):
            word_emb = np.asarray(np.random.uniform(size=(word_dim,),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
            sen.append(word_emb)
        seq_word.append(sen)
    seq_word_x = np.asarray(seq_word, dtype=theano.config.floatX)


    # variable prepare
    input_var = T.dmatrix('input_var')
    y_var = T.imatrix('y_var')
    rng = np.random.RandomState(54321)
    # real value y prepare
    y_pred_var = T.dmatrix("y_pred_var")
    y_true_var = T.imatrix("y_true_var")
    #pdb.set_trace()
    seq_len = 5
    n_output = 3
    y_true = np.asarray(np.zeros((seq_len, n_output)), dtype=np.int32)
    for item in y_true:
        item[0] = 1

    y_pred = np.asarray(np.random.uniform(size=(seq_len, n_output),
                                          low=0., high=1.),
                                          dtype=theano.config.floatX)

    return (input_var, y_true_var, y_pred_var, seq_x, seq_word_x, y_true, y_pred)
