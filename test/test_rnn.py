## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys

sys.path.append("../src")
import rnn
import test_cnn

def test_rnn():

    # parameters prepare
    seq_len = 5
    n_input = 900
    n_hidden = 300
    n_output = 3
    
    # sequence data prepare
    seq = []    
    for i in xrange(seq_len):
        sen_emb = np.asarray(np.random.uniform(size=(n_input,),
                             low=-.01, high=.01),
                             dtype=theano.config.floatX)
        seq.append(sen_emb)
    seq_x = np.asarray(seq, dtype=theano.config.floatX)

    # variable prepare
    input_var = T.dmatrix('input_var')
    y_var = T.imatrix('y_var')
    rng = np.random.RandomState(54321)
    # real value y prepare
    real_y = np.asarray(np.zeros((seq_len, n_output)), dtype=np.int32)
    for item in real_y:
        item[0] = 1
    
    rnn_model = rnn.RNN(rng,
                        input_var,
                        n_input,
                        n_hidden,
                        n_output)

    cost_var = rnn_model.loss(y_var, rnn_model.p_y_given_x_var)
    compute_loss_fn = theano.function(inputs=[input_var, y_var],
                                      outputs=[cost_var],
                                      mode="DebugMode")
    
    cost = compute_loss_fn(seq_x, real_y) # example: [array(0.6847332)]
    print "the cost of rnn is %f" %(cost[0])
    print "[RNN Test OK!]"
    return


if __name__ == "__main__":
    test_rnn()
