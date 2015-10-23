## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys

sys.path.append("../src")
import data_process


microblog_dir_path = "/Users/neutronest/projects/IDRC/rnn-sunsang/sentinet/data/weibo/fold_data/"

def test_load_microblog():
    data_process.load_microblogdata(microblog_dir_path,
                                    (1,2,3),
                                    4,
                                    5)
    return

if __name__ == "__main__":
    test_load_microblog()
