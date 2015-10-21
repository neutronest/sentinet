## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys
sys.path.append("../src")
import cnn

def test_cnn():

    x_var = T.dtensor3('x_var')
    rng = np.random.RandomState(54321)
    n_feature_maps = 300
    window_sizes = (2, 3, 4)
    dim = 300
    cnn_model = cnn.CNN(x_var, rng, dim, n_feature_maps, window_sizes)

    cnn_output_fn = theano.function(inputs=[x_var], outputs=cnn_model.output)

    # define each sentence matrix with fixed word vector and fixed sentence length
    seq_len = 5
    sen_len = 10
    seq = []
    for i in xrange(seq_len):
        sen_emb = np.asarray(np.random.uniform(size=(sen_len, dim),
                             low=-.01, high=.01),
                             dtype=theano.config.floatX)
        seq.append(sen_emb)
    seq_x = np.asarray(seq, dtype=theano.config.floatX)
    output = cnn_output_fn(seq_x)
    assert(output.shape[0] == seq_len)
    assert(output.shape[1] == n_feature_maps*len(window_sizes))
    print "[CNN Test OK!]"
    return cnn

if __name__ == "__main__":
    test_cnn()
