# -*- coding: utf-8 -*-
import pdb
import numpy as np
import theano, theano.tensor as T
import random
from collections import OrderedDict
import logging

import models
import optimizer
import utils
import config



def run_twt_experiment(load_data,
                       model,
                       model_name,
                       batch_size,
                       n_epochs,
                       optimizer_method,
                       valid_frequency=200):
    """
    the twitter experiment
    """
    logging.info("[=== begin the twitter experiment ===]")
    (train_data, valid_data, test_data) = load_data

    logging.info("[=== define cost and error variable ===]")
    y_true_var = T.imatrix("y_true_var")
    y_label_var = T.ivector("y_label_var")
    cost_train_var = model.loss(y_true_var, model.y_drop_pred)
    cost_var = model.loss(y_true_var, model.y_pred)
    error_var = model.error(y_label_var, model.output)

    logging.info("[=== define optimizer and grad variable  ==]")
    opt = optimizer.ADADELTA(model.params)
    gparams_var_list = T.grad(cost_train_var, model.params)

    fn_loss_vars = [model.input_var, y_true_var, model.mask, model.h0, model.c0, model.sh0, model.sc0, model.yt, model.yt_pred, model.if_train_var]
    fn_error_vars = []

    logging.info("[=== define ===]")

    return
