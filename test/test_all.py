## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys
sys.path.append("../src")
import cnn

import test_rnn
import test_cnn
import test_rcnn
import test_utils


if __name__ == "__main__":
    test_rnn.test_rnn()
    test_cnn.test_cnn()
    test_rcnn.test_rcnn()
    test_utils.test_binary_loss()
    #test_utils.test_nll_multiclass()