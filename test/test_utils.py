import theano, theano.tensor as T
import numpy as np
import sys
import pdb

sys.path.append("../src")
import loss

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


if __name__ == "__main__":
    test_binary_loss()
    test_nll_multiclass()