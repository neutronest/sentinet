## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys

sys.path.append("../src")
import data_process


microblog_dir_path = "../data/weibo/fold_data/"

def test_load_microblog():
    (train_x, train_y, valid_x, valid_y, test_x, test_y) = data_process.load_microblogdata(microblog_dir_path, (0, 1, 2), 3, 4)
    pdb.set_trace()
    return

if __name__ == "__main__":
    test_load_microblog()
