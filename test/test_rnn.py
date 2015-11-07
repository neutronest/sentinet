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
import utils

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
    print "[Test RNN OK!]"
    print "====="
    return

def test_rnn_onestep():
    """ test the RNN_OntStep
    """

    input_var = T.dvector()
    y_pre_var = T.dvector()
    y_var = T.ivector('y_var')
    n_input = 10
    n_hidden = 50
    n_output = 3
    h_tm1 = T.dvector('h_tm1')

    rnn_onestep_model = rnn.RNN_OneStep(input_var,
                                        y_pre_var,
                                        n_input,
                                        n_hidden,
                                        n_output,
                                        h_tm1)
    rnn_onestep_model.build_network()
    cost_var = rnn_onestep_model.loss(y_var,
                                      rnn_onestep_model.y_pred)
    sen_emb = utils.ndarray_uniform((10,), 0.05)
    h_0 = utils.ndarray_uniform((50,), 0.05)
    y_true = np.asarray([0, 1, 0], dtype=np.int32)
    y_pre = np.asarray([1, 0, 0], dtype=np.int32)
    compute_cost_fn = theano.function(inputs=[input_var, y_var, y_pre_var, h_tm1],
                                      outputs=[cost_var, rnn_onestep_model.h])

    [cost, h_current] = compute_cost_fn(sen_emb, y_true, y_pre, h_0)
    print "the cost of rnn_onestep is %f"%(cost)
    print "the current h of rnn_onestep is :"
    print h_current
    print "[Test RNN_OneStep OK!]"
    print "====="


def test_srnn():
    """
    """
    n_input = 10
    n_hidden = 50
    n_output = 3
    input_var = T.dmatrix('input_var')
    sens_pos_ndarr = np.asarray([[0, 2], [3,7], [8,10]], dtype=np.int32)
    sens_pos = theano.shared(sens_pos_ndarr)
    sens = utils.ndarray_uniform((10, 10),
                                 dtype=theano.config.floatX)


    srnn_model = rnn.SRNN(input_var,
                          n_input,
                          n_hidden,
                          n_output)
    srnn_model.build_network()
    # print srnn_model.hidden_states.eval()
    get_hidden_states_fn = theano.function(inputs=[srnn_model.input_var,
                                                   srnn_model.sens_pos_var],
                                           outputs=srnn_model.hidden_states_var)
    hs = get_hidden_states_fn(sens, sens_pos_ndarr)
    print hs
    print "[Test SCNN OK!]"
    return

def test_trnn():
    """
    """
    srnn_input = 10
    srnn_hidden = 50
    srnn_output = 3
    trnn_hidden = 50
    trnn_output = 3

    input_var = T.dmatrix('input_var')
    sens_pos_ndarr = np.asarray([[0, 2], [3,7], [8,10]], dtype=np.int32)
    sens_pos = theano.shared(sens_pos_ndarr)
    sens = utils.ndarray_uniform((10, 10),
                                 dtype=theano.config.floatX)
    relations = np.asarray([[0, -1],
                            [1, 0],
                            [2, 0]], dtype=np.int32)

    tree = np.asarray(np.zeros((trnn_hidden*(len(relations)+1),),
                               dtype=theano.config.floatX))

    srnn_model = rnn.SRNN(input_var,
                          srnn_input,
                          srnn_hidden,
                          srnn_output)
    srnn_model.build_network()

    input_t_var = T.dmatrix('input_t_var')
    trnn_model = rnn.TRNN(input_t_var,
                          srnn_hidden,
                          trnn_hidden,
                          trnn_output)
    trnn_model.build_network()
    get_srnn_h_fn = theano.function(inputs=[srnn_model.input_var,
                                            srnn_model.sens_pos_var],
                                    outputs=srnn_model.hidden_states_var)
    get_trnn_h_fn = theano.function(inputs=[trnn_model.input_var,
                                            trnn_model.relation_pairs,
                                            trnn_model.th],
                                    outputs=trnn_model.y_pred)
    hs = get_srnn_h_fn(sens, sens_pos_ndarr)
    print hs

    ts = get_trnn_h_fn(hs, relations, tree)
    print ts
    print "[Test TRNN OK!]"
    return

def test_srnn_trnn():
    """
    """
    srnn_input = 10
    srnn_hidden = 50
    srnn_output = 3
    trnn_hidden = 50
    trnn_output = 3

    input_var = T.dmatrix('input_var')
    sens_pos_ndarr = np.asarray([[0, 2], [3,7], [8,10]], dtype=np.int32)
    sens_pos = theano.shared(sens_pos_ndarr)
    sens = utils.ndarray_uniform((10, 10),
                                 dtype=theano.config.floatX)
    relations = np.asarray([[0, -1],
                            [1, 0],
                            [2, 0]], dtype=np.int32)
    tree_states = np.asarray(np.zeros((4, trnn_hidden),
                                      dtype=theano.config.floatX))
    srnn_trnn_model = rnn.SRNN_TRNN(inupt_var,
                                    srnn_input,
                                    srnn_hidden,
                                    srnn_output,
                                    trnn_input,
                                    trnn_hidden,
                                    trnn_output)

    print "[Test SRNN-TRNN OK!]"
    return
if __name__ == "__main__":
    #test_rnn()
    #test_rnn_onestep()
    test_srnn()
    test_trnn()
