import theano, theano.tensor as T
import numpy as np
import sys
import pdb

sys.path.append("../src")
import loss
import utils

def test_nll_multiclass():

    y_pred_var = T.dmatrix("y_pred_var")
    y_true_var = T.imatrix("y_true_var")
    #pdb.set_trace()
    cost_var = loss.nll_multiclass(y_true_var, y_pred_var)

    seq_len = 5
    n_output = 3
    y_true = np.asarray(np.zeros((seq_len, n_output)), dtype=np.int32)
    for item in y_true:
        item[0] = 1

    y_pred = np.asarray(np.random.uniform(size=(seq_len, n_output),
                                          low=0., high=1.),
                                          dtype=theano.config.floatX)

    cost_fn = theano.function(inputs=[y_true_var, y_pred_var],
                              outputs=[cost_var])
    cost = cost_fn(y_true, y_pred)
    print cost

    print "[Test nll_multiclass OK!]"
    print "====="
    #print "[Test binary OK!]"
    return

def test_binary_loss():

    y_true_var = T.dmatrix("y_true_var")
    y_pred_var = T.dmatrix("y_pred_var")
    cost_var = loss.binary_crossentropy(y_true_var, y_pred_var)

    seq_len = 5
    n_output = 3
    y_true = np.asarray(np.zeros((seq_len, n_output)), dtype=np.int32)
    for item in y_true:
        item[0] = 1

    y_pred = np.asarray(np.random.uniform(size=(seq_len, n_output),
                                          low=0, high=1),
                                          dtype=theano.config.floatX)

    cost_fn = theano.function(inputs=[y_true_var, y_pred_var],
                              outputs=[cost_var])

    cost = cost_fn(y_true, y_pred)

    print cost
    assert(len(cost) == 1)
    print "[Test binary loss OK!]"
    print "====="
    return

def test_mask():
    """
    """
    x_var = T.dtensor3('x_var')
    mask_matrix = utils.get_mask(x_var, 0)
    res_var = utils.get_var_with_mask(x_var, 0)

    mask_fn = theano.function(inputs=[x_var],
                              outputs=res_var)


    # generate data
    seq_len = 3
    sen_len = 5
    word_dim = 3

    seq = []
    for i in xrange(seq_len):
        sen = []
        for j in xrange(sen_len):
            sen.append([1] * word_dim)
        sen.append([0] * word_dim)
        seq.append(sen)

    seq_x = np.asarray(seq, dtype=theano.config.floatX)
    print seq_x
    print "--------- next after mask ---------------"
    seq_x_mask = mask_fn(seq_x)
    print seq_x_mask
    print "[Test mask OK!]"


    return



def test_error():
    y_true_var = T.ivector('y_true_var')
    y_pred_var = T.ivector('y_pred_var')
    y_true = [1, 2, 0, 1]
    y_pred = [0, 0, 0, 0]
    error_var = loss.mean_classify_error(y_true_var, y_pred_var)
    fn = theano.function(inputs=[y_true_var,
                                 y_pred_var],
                           outputs=error_var)
    res = fn(y_true, y_pred)
    print res
    pdb.set_trace()
    return

def test_shared_orthogonal():

    param_init = utils.shared_orthogonal((10, 10),
                                         scale=1.1,
                                         dtype=theano.config.floatX,
                                         name='W')
    print param_init.get_value()
    return

if __name__ == "__main__":
    #test_binary_loss()
    #test_mask()
    #test_nll_multiclass()
    #test_error()
    test_shared_orthogonal()
