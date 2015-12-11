# -*- coding: utf-8 -*-
import sys, getopt
import pdb
import rcnn
import models
import optimizer
import data_process
import logging
import theano, theano.tensor as T
import theano.typed_list
import numpy as np
import random
from collections import OrderedDict
import utils
from theano import ProfileMode


def check_process(check_idx,
                  model,
                  data_x,
                  data_y,
                  loss_fn,
                  error_fn,
                  process_type):

    check_num = 0
    check_loss_sum = 0.
    check_loss_res = 0.
    check_error_sum = 0.
    check_error_res = 0.

    polarity_n = [0, 0, 0]

    sen_num = 0.
    logging.info("=== check process %d==="%(check_idx))
    check_idx += 1
    for(check_threadid_x, check_item_x), (check_threadid_y, check_item_y) in \
       zip(data_x.items(), data_y.items()):
        assert(check_threadid_x == check_threadid_y)

        input_x = np.asarray(check_item_x[0],
                                dtype=np.int32)

        input_y = np.asarray([ [1 if i == y else 0 for i in xrange(3)]  for y in check_item_y],
                             dtype=np.int32)
        label_y = np.asarray(check_item_y,
                             dtype=np.int32)
        [check_loss, check_output] = loss_fn(input_x, input_y, 0)
        [check_error, check_output] = error_fn(input_x, label_y, 0)
        for p in check_output:
            polarity_n[p] += 1
        check_loss_sum += (check_loss / len(label_y))
        check_num += 1
        check_error_sum += sum([i for i in check_error if i == 1])
        sen_num += len(label_y)
    # caculate the result
    check_loss_res = check_loss_sum / check_num
    check_error_sum = check_error_sum / sen_num
    logging.info("the %d's %s loss is %f"%(check_idx, process_type, check_loss_res))
    logging.info("the %d's %s error is %f"%(check_idx, process_type, check_error_sum))
    logging.info("%s pred polarity distribute: %d %d %d"%(process_type,
                                                          polarity_n[0],
                                                           polarity_n[1],
                                                           polarity_n[2]))
    return (check_loss_res, check_error_sum)


if __name__ == "__main__":
    options, args = getopt.getopt(sys.argv[1:], "",
        ["help",
         "experiment=",
         "model_name=",
         "dataset_name=",
         "log_path=",
         "word_dim=",
         "level1_model_name=",
         "level2_model_name=",
         "level1_input=",
         "level1_hidden=",
         "level2_input=",
         "level2_hidden=",
         "cnn_n_feature_maps=",
         "cnn_window_sizes=",
         "n_output=",
         "if_dropout=",
         "optimizer_method=",
         "learning_rate=",
         "batch_size=",
         "n_epochs=",
         "train_pos=",
         "valid_pos=",
         "test_pos=",
         "valid_frequency="])

    for opt, arg in options:
        print (opt, arg)
        if opt == "--help":
            print "this is the baseline process, need some help?"
        elif opt == "--experiment":
            experiment = arg

        elif opt == "--model_name":
            model_name = arg

        elif opt == "--dataset_name":
            print arg
            dataset_name = arg

        elif opt == "--log_path":
            # example of arg: ../logs/result1.txt of str
            log_path = arg

        elif opt == "--word_dim":
            word_dim = int(arg)

        elif opt == "--level1_model_name":
            level1_model_name = arg

        elif opt == "--level2_model_name":
            level2_model_name = arg

        elif opt == "--level1_input":
            level1_input = arg

        elif opt == "--level1_hidden":
            level1_hidden = arg

        elif opt == "--level2_input":
            level2_input = arg

        elif opt == "--level2_hidden":
            level2_hidden = arg

        elif opt == "--cnn_n_feature_maps":
            # example of arg: 300 of str
            cnn_n_feature_maps = int(arg)

        elif opt == "--cnn_window_sizes":
            # example of arg: 2,3,4 of str
            cnn_window_sizes = tuple([int(t) for t in arg.split("@")])
        elif opt == "--n_output":
            n_output = int(arg)

        elif opt == "--if_dropout":
            # example of arg: 0.5 of str
            if_dropout = arg

        elif opt == "--optimizer_method":
            optimizer_method = arg

        elif opt == "--learning_rate":
            # example of arg: 0.01 of str
            learning_rate = float(arg)

        elif opt == "--batch_size":
            # example of arg: 10 of str
            batch_size = int(arg)

        elif opt == "--n_epochs":
            # example of arg: 100 of str
            n_epochs = int(arg)

        elif opt == "--train_pos":
            # example of arg: 1000 of str
            if '@' not in arg:
                train_pos = int(arg)
            else:
                train_pos = tuple([int(t) for t in arg.split("@")])
        elif opt == "--valid_pos":
            # example of arg: 1005 of str
            valid_pos = int(arg)

        elif opt == "--test_pos":
            test_pos = int(arg)
        elif opt=="--valid_frequency":
            valid_frequency = int(arg)
            # define log file
    print "prepare the logging file"
    assert(log_path != None)
    logging.basicConfig(
        level=logging.DEBUG, format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', filename=log_path, filemode='w')
    words_table, lookup_table, wordid_acc = data_process.build_lookuptable()
    logging.info("words_table len: %d"%(len(words_table)))
    logging.info("lookup_table len: %d"%(len(lookup_table)))
    logging.info("wordid_acc: %d"%(wordid_acc))
    logging.info("loading microblog data now!")
    load_data = data_process.load_microblogdata(train_pos, valid_pos, test_pos, words_table)
    x_var =T.imatrix('x_var')
    if model_name == "cnn_model":
        model = models.SingleModel(model_name,
                                   x_var,
                                   lookup_table,
                                   "none",
                                   "none",
                                   n_output,
                                   if_dropout,
                                   word_dim,
                                   cnn_n_feature_maps,
                                   cnn_window_sizes)
    else:
        model = models.Model(level1_model_name,
                             level2_model_name,
                             x_var,
                             lookup_table,
                             level1_input,
                             level1_hidden,
                             level2_input,
                             level2_hidden,
                             n_output,
                             word_dim,
                             cnn_n_feature_maps,
                             cnn_window_sizes,
                             if_dropout)
        print "TODO: CNN-RNN model"
    logging.info("begin cnn method")
    # prepare data
    (train_x, train_y, valid_x, valid_y, test_x, test_y) = load_data


    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    logging.info("[==== Data Size description]")
    logging.info("[the train seqs size is %d]"%(n_train))
    logging.info("[the valid seqs size is %d]"%(n_valid))
    logging.info("[the test seqs size is %d]"%(n_test))
    # DEFINE VARIABLE
    logging.info("%s experiment began!]"%(model_name))
    y_true_var = T.imatrix('y_true_var')
    y_label_var = T.ivector('y_label_var')

    cost_train_var = model.loss(y_true_var, model.y_drop_pred)
    cost_var = model.loss(y_true_var, model.y_pred)
    error_var = model.error(y_label_var, model.output)
    # optimizer define
    logging.info("[minibatch used]")
    logging.info("[optimizer %s define!]"%(optimizer_method))
    if optimizer_method == "sgd":
        opt = optimizer.SGD(learning_rate=learning_rate)
        opt.delta_pre_init(model.params)
    if optimizer_method == "adadelta":
        opt = optimizer.ADADELTA(model.params)

    gparams_var_list = T.grad(cost_train_var, model.params)
    fn_loss_vars = [model.input_var, y_true_var, model.if_train_var]
    fn_error_vars = [model.input_var, y_label_var, model.if_train_var]
    compute_gparams_fn = theano.function(inputs=fn_loss_vars,
                                             outputs=gparams_var_list,
                                             on_unused_input='ignore')

    train_loss_fn = theano.function(inputs=fn_loss_vars,
                                        outputs=[cost_train_var,
                                                 model.y_pred],
                                        on_unused_input='ignore')

    compute_loss_fn = theano.function(inputs=fn_loss_vars,
                                          outputs=[cost_var,
                                                   model.y_pred],
                                          on_unused_input='ignore')
    compute_error_fn = theano.function(inputs=fn_error_vars,
                                       outputs=[error_var,
                                                model.output],
                                       on_unused_input='ignore')

    epoch = 0
    seq_idx = 0
    valid_idx = 0
    test_idx = 0
    epoch = 0
    train_num = 0
    train_loss_res = 0.
    train_loss_sum = 0.
    train_error_sum = 0
    sen_num = 0
    polarity_train_n = [0, 0, 0]
    early_stopping_val = 999
    valid_check_list = []
    test_error_list = []
    final_error = 0.

    logging.info("=== Begin to Train! ===")
    while epoch < n_epochs:
        logging.info("[===== EPOCH %d BEGIN! =====]" %(epoch))
        print "began to train epoch %d"%(epoch)
        seq_idx = 0
        epoch += 1
        for (train_threadid_x, train_item_x), (train_threadid_y, train_item_y) in \
            zip(train_x.items(), train_y.items()):

            input_x = np.asarray(train_item_x[0],
                                 dtype=np.int32)
            input_y = np.asarray([ [1 if i == y else 0 for i in xrange(3)]  for y in train_item_y],
                                 dtype=np.int32)

            label_y = np.asarray(train_item_y,
                                 dtype=np.int32)
            g = compute_gparams_fn(input_x, input_y, 1)
            [train_loss, y] = train_loss_fn(input_x, input_y, 1)
            [train_error, y] = compute_error_fn(input_x, label_y, 1)
            opt.gparams_update(g)
            train_loss /= len(label_y)
            train_loss_sum += train_loss
            train_error_sum += sum([i for i in train_error if i == 1])
            sen_num += len(label_y)
            train_num += 1
            seq_idx += 1
            for p in y:
                polarity_train_n[p] += 1

            if seq_idx % batch_size == 0:
                # update the params
                opt.params_update(model.params)
                train_loss_res = train_loss_sum / train_num
                logging.info("[=== batch train loss: %f ===]"%(train_loss_res))
                train_error_res = train_error_sum * 1. / sen_num

                # reinit
                train_num = 0
                train_loss_sum = 0.
                train_loss_res = 0.
                train_error_sum = 0.
                sen_num = 0
                polarity_train_n = [0, 0, 0]
            if seq_idx % valid_frequency == 0:
                ### VALID PROCESS
                logging.info("[VALID PROCESS]")
                (loss_res, error_res) = check_process(valid_idx,
                                                      model,
                                                      valid_x,
                                                      valid_y,
                                                      compute_loss_fn,
                                                      compute_error_fn,
                                                      "valid")
                valid_check_list.append(loss_res)
                valid_idx += 1
                # TEST PROCESS
                logging.info("[TEST PROCESS]")
                (_, test_error_res) = check_process(test_idx,
                                                    model,
                                                    test_x,
                                                    test_y,
                                                    compute_loss_fn,
                                                    compute_error_fn,
                                                    "test")
                test_idx += 1
                test_error_list.append(test_error_res)
                if len(valid_check_list) == 30:
                    min_loss = min(valid_check_list)
                    if early_stopping_val >= min_loss:
                        early_stopping_val = min_loss
                        res_id = valid_check_list.index(min_loss)
                        final_error = test_error_list[res_id]
                        valid_check_list = []
                        test_error_list = []
                        logging.info("[=== choosed valid loss: %f ===]"%(early_stopping_val))
                        logging.info("[=== choosed test error: %f ===]"%(final_error))
                    else:
                        logging.info("[=== early stopping! ===]")
                        logging.info("[=== final valid loss: %f ===]"%(early_stopping_val))
                        logging.info("[=== final test error: %f ===]"%(final_error))
                        exit()
    exit()
