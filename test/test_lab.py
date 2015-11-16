# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
import pdb
import pydot
import sys, os
sys.path.append("../src/")
import utils
# theano.config.exception_verbosity = "high"

def test_1():

    a = None
    if a == None:
        print "1"
    if a == None:
        print "2"
    return


def test_2():

    x_var = T.dvector('x_var')
    y_var = T.dvector('y_var')
    pairs_var = T.imatrix('pairs_var')

    def _op(pair, a, b):
        return a.take(pair[0]) + b.take(pair[1])
    res_var, _ = theano.scan(fn=_op,
                             sequences=[pairs_var],
                             non_sequences=[x_var, y_var],
                             outputs_info=None)

    get_res_fn = theano.function(inputs=[x_var, y_var, pairs_var],
                                 outputs=res_var)

    x = np.asarray([10, 20, 30, 40], dtype=theano.config.floatX)
    y = np.asarray([1,2, 3, 4], dtype=theano.config.floatX)
    pairs = np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
    res = get_res_fn(x, y, pairs)
    print res
    return

def test_4():
    x_var = T.dmatrix('x_var')
    #x = np.asarray([i for i in xrange(0, 50)],
    #               dtype=theano.config.floatX)
    x = np.asarray([[i*10 for i in xrange(j,j+3)] for j in xrange(0,50,3)],
                   dtype=theano.config.floatX)
    print x
    pairs_var = T.imatrix('pairs_var')
    pairs = np.asarray([[0, 4],[5,9]],
                       dtype=np.int32)

    def _op(pair):
        #return x_var.take(T.arange(pair[0], pair[1]+1))
        return x_var[pair[0]:(pair[1]+1)]

    res_var, _  = theano.scan(fn=_op,
                              sequences=pairs_var,
                              outputs_info=None)
    get_res_fn = theano.function(inputs=[x_var, pairs_var],
                                 outputs=res_var)
    res = get_res_fn(x, pairs)
    print res
    return

class S(object):
    def __init__(self):
        self.x_var = theano.shared(np.asarray([1], dtype=np.int32), name='x')
        return
    def update(self):
        self.x_var = self.x_var + 1
        return

def test_3():

    s1 = S()
    s1.update()
    s1.update()
    print s1.x_var.eval()
    s2 = S()
    s2.update()
    print s2.x_var.eval()
    return


def test_sth():
    """
    """
    x_var = T.ivector('x_var')
    subx_var = x_var.take([i for i in xrange(0,4)])

    get_sub_fn = theano.function(inputs=[x_var],
                           outputs=subx_var)
    subx = get_sub_fn(np.asarray([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int32))
    print subx # hope [2, 3, 4, 5]
    return

def test_6():
    z = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    y = np.asarray([5, 6], dtype=np.int32)
    z_var = T.imatrix('z_var')
    x_var = z_var + 1
    y_var = T.ivector('y_var')
    x_var = T.set_subtensor(x_var[0, :], y_var)
    fn = theano.function(inputs=[z_var, y_var],
                         outputs=x_var)

    x_new = fn(z, y)
    print x_new
    return

def test_7():
    x = theano.shared(np.asarray([[1, 2], [3, 4], [5, 6]],
                 dtype=theano.config.floatX))
    pos_var = T.imatrix('pos_var')

    def _op(pos):
        return x[pos[0]][pos[1]]

    r, _ = theano.scan(fn=_op,
                       sequences=pos_var,
                       outputs_info=None)
    get_res_fn = theano.function(inputs=[pos_var],
                                 outputs=r)
    pos_val = np.asarray([[0,1], [1,0], [2,1]],
                         dtype=np.int32)
    print get_res_fn(pos_val)
    return

def test_8():
    a = theano.tensor.vector("a")      # declare symbolic variable
    b = a + a ** 10                    # build symbolic expression
    f = theano.function([a], b)
    theano.printing.pydotprint(b,
                               outfile="./symbolic_graph_unopt.png",
                               format="png",
                               var_with_name_simple=True)

    return

def test_9():
    x_var = T.dmatrix('x_var')
    y_var = T.dmatrix('y_var')

    x_new_var = T.concatenate([x_var, y_var])
    fn = theano.function(inputs=[x_var, y_var],
                         outputs=x_new_var)

    x = np.asarray([[0, 0, 0], [1, 1, 1]],
                   dtype=theano.config.floatX)
    y = np.asarray([[2,2,2]],
                   dtype=theano.config.floatX)
    res = fn(x, y)
    print res
    return


def test_10():
    x_var = T.dmatrix('x_var')
    y_var = T.dmatrix('y_var')

    def _op(y, h_tm1):
        h = theano.shared(np.asarray([1, 1, 1], dtype=theano.config.floatX))
        h_next = T.concatenate([h_tm1, h.dimshuffle('x', 0)])
        return h_next

    h_res, _ = theano.scan(fn=_op,
                           sequences=y_var,
                           outputs_info=x_var)

    fn = theano.function(inputs=[x_var, y_var],
                         outputs=h_res)

    x = np.asarray([[0, 0, 0]],
                   dtype=theano.config.floatX)
    y = np.asarray([[1,1,1], [2,2,2], [3,3,3]],
                   dtype=theano.config.floatX)
    res = fn(x, y)
    print res
    return

def test_11():
    h = theano.shared(np.asarray([1, 1, 1], dtype=theano.config.floatX))
    h_tm = h.dimshuffle('x', 0)
    print h_tm.eval()
    return

def test_12():

    x = T.ivector('x')
    h0 = T.scalar('h0')

    def _op(x, h):
        y = h + x
        return y

    r, _ = theano.scan(fn=_op,
                       sequences=x,
                       outputs_info=h0)

    xlist = np.asarray([1,2,3],
                       dtype=np.int32)
    h = 1
    fn = theano.function(inputs=[x, h0],
                         outputs=r)
    res = fn(xlist, h)
    print res
    return

def test_13():
    x_var = T.dmatrix('x_var')
    x_new = T.flatten(x_var)
    fn = theano.function(inputs=[x_var],
                         outputs=x_new)
    x = np.asarray([[1,2,3],[4,5,6]],
                   dtype=theano.config.floatX)
    res = fn(x)
    print res
    return

def test_14():
    x_var = T.dmatrix('x_var')
    pos_var = T.dmatrix('pos_var')
    # TODO
    return

if __name__ == "__main__":
    #test_1()
    #test_2()
    #test_3()
    #test_sth()
    #test_4()
    #test_10()
    test_13()
