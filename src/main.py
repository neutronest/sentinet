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


"""
======================= MICROBLOG EXPERIMENT ===========
"""

def run_microblog_experimentV2(load_data,
                               model,
                               model_name,
                               batch_type,
                               batch_size,
                               n_epochs,
                               valid_frequency,
                               learning_rate,
                               optimizer_method="sgd"):
    """ the microblog experimentV2


    """
    # prepare data
    (train_x, train_y, valid_x, valid_y, test_x, test_y) = load_data

    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    logging.info("[==== Data Size description]")
    logging.info("[the train seqs size is %d]"%(n_train))
    logging.info("[the valid seqs size is %d]"%(n_valid))
    logging.info("[the test seqs size is %d]"%(n_test))

    if model_name == "srnn_trnn_model":
        # DEFINE VARIABLE
        logging.info("[srnn-trnn model experiment began!]")
        y_true_var = T.imatrix('y_true_var')
        y_label_var = T.ivector('y_label_var')
        cost_var = model.loss(y_true_var, model.y_pred)
        error_var = model.error(y_label_var, model.output)

        #theano.printing.pydotprint(cost_var, outfile="./graph.png", var_with_name_simple=True)
        # optimizer define
        logging.info("[minibatch used]")
        logging.info("[optimizer define!]")
        opt = optimizer.SGD(learning_rate=learning_rate)
        opt.delta_pre_init(model.params)
        gparams_var_list = T.grad(cost_var, model.params)

        compute_gparams_fn = theano.function(inputs=[model.input_var,
                                                     y_true_var,
                                                     model.sens_pos_var,
                                                     model.relation_pairs,
                                                     model.th],
                                             outputs=gparams_var_list)
        compute_loss_fn = theano.function(inputs=[model.input_var,
                                                  y_true_var,
                                                  model.sens_pos_var,
                                                  model.relation_pairs,
                                                  model.th],
                                          outputs=cost_var)

        compute_error_fn = theano.function(inputs=[model.input_var,
                                                   y_label_var,
                                                   model.sens_pos_var,
                                                   model.relation_pairs,
                                                   model.th],
                                           outputs=[error_var, model.output])

        epoch = 0
        seq_idx = 0
        valid_idx = 0
        epoch = 0

        train_num = 0
        train_loss_res = 0.
        train_loss_sum = 0.
        logging.info("=== Begin to Train! ===")
        while epoch < n_epochs:
            logging.info("[===== EPOCH %d BEGIN! =====]" %(epoch))
            seq_idx = 0
            epoch += 1
            for (train_threadid_x, train_item_x), (train_threadid_y, train_item_y) in \
                zip(train_x.items(), train_y.items()):
                assert(train_threadid_x == train_threadid_y)
                train_input_x = np.asarray(train_item_x[0],
                                           dtype=theano.config.floatX)
                train_input_y = np.asarray([ [1 if i == y else 0 for i in xrange(3)]  for y in train_item_y],
                                           dtype=np.int32)
                sens_pos = np.asarray(train_item_x[1],
                                      dtype=np.int32)
                relation_tree = np.asarray(train_item_x[2],
                                           dtype=np.int32)

                th_init = np.asarray(np.zeros(model.trnn_model.n_hidden*(len(relation_tree)+1)))
                #pdb.set_trace()
                g = compute_gparams_fn(train_input_x,
                                       train_input_y,
                                       sens_pos,
                                       relation_tree,
                                       th_init)
                opt.gparams_update(g)
                train_loss = compute_loss_fn(train_input_x,
                                             train_input_y,
                                             sens_pos,
                                             relation_tree,
                                             th_init)
                train_loss_sum += train_loss
                train_num += 1
                seq_idx += 1
                if seq_idx % batch_size == 0:
                    # update the params
                    opt.params_update(model.params)
                    train_loss_res = train_loss_sum / train_num
                    logging.info("train loss: %f"%(train_loss_res))
                    # reinit
                    train_num = 0
                    train_loss_sum = 0.
                    train_loss_res = 0.


                if seq_idx % valid_frequency == 0:
                    # valid
                    valid_num = 0
                    valid_loss_sum = 0.
                    valid_loss_res = 0.
                    valid_error_sum = 0.
                    valid_error_res = 0.
                    sen_num = 0.
                    logging.info("=== valid process %d==="%(valid_idx))
                    valid_idx += 1
                    for(valid_threadid_x, valid_item_x), (valid_threadid_y, valid_item_y) in \
                       zip(valid_x.items(), valid_y.items()):
                        assert(valid_threadid_x == valid_threadid_y)
                        valid_input_x = np.asarray(valid_item_x[0],
                                                   dtype=theano.config.floatX)
                        valid_input_y = np.asarray([ [1 if i == y else 0 for i in xrange(3)]  for y in valid_item_y],
                                                   dtype=np.int32)
                        valid_label_y = np.asarray(valid_item_y,
                                                   dtype=np.int32)
                        sens_pos = np.asarray(valid_item_x[1],
                                              dtype=np.int32)
                        relation_tree = np.asarray(valid_item_x[2],
                                                   dtype=np.int32)
                        th_init = np.asarray(np.zeros(model.trnn_model.n_hidden*(len(relation_tree)+1)))
                        valid_loss = compute_loss_fn(valid_input_x,
                                                     valid_input_y,
                                                     sens_pos,
                                                     relation_tree,
                                                     th_init)
                        [valid_error, valid_output] = compute_error_fn(valid_input_x,
                                                                       valid_label_y,
                                                                       sens_pos,
                                                                       relation_tree,
                                                                       th_init)
                        valid_loss_sum += valid_loss
                        valid_num += 1
                        valid_error_sum += sum([i for i in valid_error if i == 1])
                        sen_num += len(relation_tree)
                    # caculate the result
                    valid_loss_res = valid_loss_sum / valid_num
                    valid_error_sum = valid_error_sum / sen_num
                    logging.info("the %d's valid loss is %f"%(valid_idx, valid_loss_res))
                    logging.info("the %d's valid error is %f"%(valid_idx, valid_error_sum))

    return

"""=================================================="""


def run_microblog_experiment(load_data,
                             model,
                             model_name,
                             batch_type,
                             batch_size,
                             n_epochs,
                             valid_frequency,
                             learning_rate,
                             optimizer_method="sgd"):
    """ the microblog experiment


    """
    # prepare data
    (train_x, train_y, valid_x, valid_y, test_x, test_y) = load_data

    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    logging.info("[==== Data Size description]")
    logging.info("[the train seqs size is %d]"%(n_train))
    logging.info("[the valid seqs size is %d]"%(n_valid))
    logging.info("[the test seqs size is %d]"%(n_test))

    # CHOOSE MODEL
    if model_name == "rcnn_onestep_model":
        """
        ------------------  rcnn_onestep_model ------------------
        """
        # DEFINE VARIABLE
        logging.info("[rcnn_onestep_model experiment began!]")
        y_true_var = T.ivector('y_true_var')
        y_label_var = T.scalar('y_label_var')
        h_pre_var = model.h_pre_var
        cost_var = model.loss(y_true_var, model.y_pred)
        error_var = model.error(y_label_var, model.output_var)


        # optimizer define
        logging.info("[minibatch used]")
        logging.info("[optimizer define!]")
        opt = optimizer.SGD(learning_rate=learning_rate)
        opt.delta_pre_init(model.params)

        gparams_var_list = T.grad(cost_var, model.params)

        compute_gparams_fn = theano.function(inputs=[model.input_var,
                                                     y_true_var,
                                                     model.h_pre_var],
                                             outputs=gparams_var_list)
        compute_loss_fn = theano.function(inputs=[model.input_var,
                                                  y_true_var,
                                                  model.h_pre_var],
                                          outputs=[cost_var, model.h])
        compute_error_fn = theano.function(inputs=[model.input_var,
                                                   y_label_var,
                                                   model.h_pre_var],
                                           outputs=[error_var, model.output_var])
        seq_idx = 0
        valid_idx = 0
        epoch = 0

        train_sen_num = 0
        train_loss_res = 0.
        train_loss_sum = 0.
        train_error = 0.
        train_recall = [0, 0, 0]
        while epoch < n_epochs:
            logging.info("[===== EPOCH %d BEGIN! =====]" %(epoch))
            seq_idx = 0
            epoch += 1
            for (train_kx, train_vx), (train_ky, train_vy) in zip(train_x.items(), train_y.items()):
                # EACH THREAD
                #logging.info("---next thread---")
                assert(train_kx == train_ky)
                threadid = train_kx
                train_sen_num += len(train_vx)
                # we must store the each node's hidden vector
                h_state = {}
                h_state['-1'] = utils.ndarray_uniform((model.rnn_hidden), 0.01, theano.config.floatX)
                for item_x, item_y in zip(train_vx, train_vy):
                    # get each sentence
                    (docid_x, parent, words_emb) = item_x
                    (docid_y, label) = item_y
                    ### TODO: DEBUG
                    assert(docid_x == docid_y)
                    """
                    if h_state.get(docid_x) == None:
                        h_state[docid_x] = utils.ndarray_uniform((model.rnn_hidden), 0.01, theano.config.floatX)
                    """
                    # prepare input
                    input_x = words_emb
                    input_y = np.asarray([1 if i == label else 0  for i in xrange(3)], dtype=np.int32)
                    h_pre = h_state[str(parent)]
                    # update gradients
                    g = compute_gparams_fn(input_x, input_y, h_pre)
                    opt.gparams_update(g)
                    # compute train loss
                    [train_loss, h]= compute_loss_fn(input_x, input_y, h_pre)
                    h_state[str(docid_x)] = h
                    train_loss_sum += train_loss
                    # compute train error
                    [error, label_pred] = compute_error_fn(input_x, label, h_pre)
                    label_pred = int(label_pred)
                    train_error += error
                    train_recall[label_pred] += 1
                    #logging.info("the train loss of train idx %d is: %f" %(seq_idx, train_loss))
                # END one thread training
                seq_idx += 1
                if seq_idx % batch_size == 0:
                    # UPDATE PARAMS
                    #logging.info("[update the params at epoch %d, seq %d]"%(epoch, seq_idx))
                    train_loss_res = train_loss_sum / train_sen_num
                    train_error = train_error * 1. / train_sen_num
                    logging.info("[the train loss of %d is: %f]"%(seq_idx, train_loss_res))
                    logging.info("[the train error is: %f]"%(train_error))
                    logging.info("the predict num is %d, %d, %d"%(train_recall[0], train_recall[1], train_recall[2]))
                    # re-init
                    train_error = 0
                    train_recall = [0, 0, 0]
                    train_loss_sum = 0
                    train_sen_sum = 0
                    opt.params_update(model.params)
                    # TODO: DEBUG
                    #print model.params[6][-1].eval()
                if seq_idx % valid_frequency == 0:
                    valid_idx += 1

                    # statistic value
                    cost_res = 0.
                    cost_sum = 0.
                    error_sum = 0.
                    error_res = 0.
                    n_cost = 0
                    valid_recall = {}
                    for i in xrange(3):
                        valid_recall[i] = 0
                    logging.info("[===== began to idx %d validation =====]"%(valid_idx))
                    # VALID PROCESS
                    for (valid_kx, valid_vx), (valid_ky, valid_vy) in zip(valid_x.items(), valid_y.items()):
                        # EACH THREAD
                        # TODO: CHECK VALID
                        # check threadid
                        assert(valid_kx == valid_ky)
                        threadid = valid_kx
                        n_sens = len(valid_vx)
                        n_cost += n_sens
                        h_state = {}
                        h_state['-1'] = utils.ndarray_uniform((model.rnn_hidden), 0.01, theano.config.floatX)
                        for item_x, item_y in zip(valid_vx, valid_vy):
                            # get each valid sentence
                            (docid_x, parent, words_emb) = item_x
                            (docid_y, label) = item_y
                            assert(docid_x == docid_y)
                            # valid data prepare
                            input_x = words_emb
                            input_y = np.asarray([1 if i == label else 0 for i in xrange(3)], dtype=np.int32)
                            h_pre = h_state[str(parent)]
                            [valid_loss, h] = compute_loss_fn(input_x, input_y, h_pre)
                            h_state[str(docid_x)] = h
                            [error, label_pred] = compute_error_fn(input_x, label, h_pre)
                            label_pred = int(label_pred)
                            valid_recall[label_pred] += 1
                            error_sum += error
                            cost_sum += valid_loss
                    cost_res = cost_sum / n_cost
                    error_res = error_sum / n_cost
                    logging.info("[IMPORTANT: the loss of valid-idx %d is: %f ======]" %(valid_idx, cost_res))
                    logging.info("[IMPORTANT: the error of valid-idx %d is: %f ======]" %(valid_idx, error_res))
                    logging.info("[IMPORTANT: the negative num of predict: %d]"%(valid_recall[0]))
                    logging.info("[IMPORTANT: the neutral num of predict: %d]"%(valid_recall[1]))
                    logging.info("[IMPORTANT: the positive num of predict: %d]"%(valid_recall[2]))
            opt.learning_rate_decay()
        """

        -------------------------------- END rcnn_onestep_model -----------------------
        """



        """
        -------------------------------- other model
        """
    return

"""
========================= SWDA EXPERIMENT ===============
"""
def run_swda_experiment(load_data,
                        model,
                        batch_type,
                        batch_size,
                        optimizer="sgd"):
    """
    the rnn experiment with SWDA dataset

    Parameters:
    -----------
    dataset: the dataset, example:
        example: (train_x, train_y, valid_x, valid_y, test_x, test_y)

    model: the deep learning model we used
        example: rcnn, rcnn_f, lstm, lstm_f

    input_var: the theano variable that contain the original input data symbol.
        type: theano.tensor
    Return

    """

    (train_x, train_y, valid_x, valid_y, test_x, test_y) = load_data
    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    epoch = 0

    # variable define
    y_var = None
    lr_var = None
    label_var = None
    cost_var = None
    error_var = None
    # optimizer define
    gparams = None
    optimizer_updates = OrderedDict()

    # standard batchsize or mini batch
    if batch_type == "all":
        logging.info("the batch type is all!")
        y_var = T.imatrix('y_var')
        lr_var = T.scalar('lr_var')
        label_var = T.vector('label_var')
        cost_var = model.loss(y_var, model.y)
        error_var = model.error(label_var, model.output_var)
    """
    elif batch_type == "minibatch":
        logging.info("the batch type is minibatch")
        y_var = [T.imatrix()] * batch_size
        label_var = [T.vector()] * batch_size
        cost_var = None
        for i in xrange(0, batch_size):
            cost_var += model.loss(model.y, y_var[i])
        lr_var = T.scalar("lr_var")
    """

    if optimizer == "sgd":
        print "using sgd for optimization"
        logging.info("using sgd for optimization")
        gparams = [T.grad(cost_var, param_var) for param_var in model.params]
        optimizer_updates = [(param, param - gparam * lr_var) \
            for param, gparam in zip(model.params, gparams)]


    #compute_gradients = theano.function(inputs=[x_var, y_var],
    #                                    outputs=gparams)
    train_loss_fn = theano.function(inputs=[model.input_var, y_var, lr_var],
                                    outputs=cost_var,
                                    updates=optimizer_updates)

    compute_loss_fn = theano.function(inputs=[model.input_var, y_var],
                                      outputs=cost_var,
                                      mode="DebugMode")
    compute_error_fn = theano.function(inputs=[model.input_var, label_var],
                                       outputs=cost_var,
                                       mode="DebugMode")

    print "begin to train"
    logging.info("begin to train")
    valid_idx = 0
    while epoch < n_epochs:
        valid_idx += 1
        epoch += 1
        #gparams_acc = None
        train_losses = 0.
        # mini batch
        logging.info("prepare the %i's mini batch"%(epoch))
        logging.info("the size of batch is %i"%(batch_size))

        batch_index_list = [x for x in xrange(0, 1000, batch_size)]

        for batch_index in batch_index_list:
            batch_start = batch_index
            batch_stop = batch_start + batch_size
            # training on each batch
            for idx in xrange(batch_start, batch_stop):
                # accumulate gradients
                train_loss = train_loss_fn(utils.wrap_x(train_x[idx]),
                                           utils.expand_y(train_y[idx], 43),
                                           learning_rate) # train_loss: list of float
                train_loss_avg = np.mean(train_loss)
                train_losses += train_loss_avg
                logging.info("the seq %i's train loss is: %f"%(idx, train_loss_avg))
                logging.info("epoch %i's train losses is: %f" %(epoch, train_losses))
        # valid process
        if valid_idx % 20 == 0:
            error_sum = 0
            item_sum = 0
            for vdx in xrange(n_valid):
                valid_loss = compute_loss_fn(utils.wrap_x(valid_x[vdx]),
                                             utils.expand_y(valid_y[vdx],43))
                valid_error = compute_error_fn(utils.wrap_x(valid_x[vdx]),
                                               utils.wrap_y(valid_y[vdx]))
                item_sum += len(valid_y[vdx])
                error_sum += valid_error
            accurate_res = 0.
            accurate_res = 1. - (error_sum  * 1. / item_sum)
            logging.info("the accurate of valid set is %f"%(accurate_res))
    return

"""
================  MAIN PROCESS =======================
"""


if __name__ == "__main__":
    """
    the main process of rnnlab
    """

    # all params need init
    experiment = None
    model_name = None
    dataset_name = None
    log_path = None
    word_dim = None
    cnn_n_feature_maps = None
    cnn_window_sizes = None
    rnn_n_hidden = None
    rnn_n_outupt =  None
    level1_input = None
    level1_hidden = None
    level1_output = None
    level2_input = None
    level2_hidden = None
    level2_output = None
    dropout_rate = None
    optimizer_method = None
    learning_rate = None
    batch_type = None
    batch_size = None
    n_epochs = None
    train_pos = None
    valid_pos = None
    test_pos = None
    run_model = None
    load_data = None

    # Get the arguments from command line
    options, args = getopt.getopt(sys.argv[1:], "",
        ["help",
         "experiment=",
         "model_name=",
         "dataset_name=",
         "log_path=",
         "word_dim=",
         "cnn_n_feature_maps=",
         "cnn_window_sizes=",
         "rnn_n_hidden=",
         "rnn_n_output=",
         "level1_input=",
         "level1_hidden=",
         "level1_output=",
         "level2_input=",
         "level2_hidden=",
         "level2_output=",
         "dropout_rate=",
         "optimizer_method=",
         "learning_rate=",
         "batch_type=",
         "batch_size=",
         "n_epochs=",
         "train_pos=",
         "valid_pos=",
         "test_pos=",
         "valid_frequency="])

    for opt, arg in options:
        print (opt, arg)
        if opt == "--help":
            print "this is the rnn model main function. Need some help?"

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

        elif opt == "--cnn_n_feature_maps":
            # example of arg: 300 of str
            cnn_n_feature_maps = int(arg)

        elif opt == "--cnn_window_sizes":
            # example of arg: 2,3,4 of str
            cnn_window_sizes = tuple([int(t) for t in arg.split("@")])

        elif opt == "--rnn_n_hidden":
            # example of arg: 300 of str
            rnn_n_hidden = int(arg)

        elif opt == "--rnn_n_output":
            # example of arg: 43 of str
            rnn_n_output = int(arg)

        elif opt == "--level1_input":
            level1_input = int(arg)
        elif opt == "--level1_hidden":
            level1_hidden = int(arg)
        elif opt == "--level1_output":
            level1_output = int(arg)

        elif opt == "--level2_input":
            level2_input = int(arg)
        elif opt == "--level2_hidden":
            level2_hidden = int(arg)
        elif opt == "--level2_output":
            level2_output = int(arg)

        elif opt == "--dropout_rate":
            # example of arg: 0.5 of str
            dropout_rate = float(arg)

        elif opt == "--optimizer_method":
            if arg == "sgd":
                optimizer_method = optimizer.SGD()
            # TODO: other optimizer

        elif opt == "--learning_rate":
            # example of arg: 0.01 of str
            learning_rate = float(arg)

        elif opt == "--batch_type":
            # example of arg: all/minibatch
            batch_type = arg

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

    """
    ========= Choose Dataset =============

    """

    # define dataset
    assert(dataset_name != None)
    assert(train_pos != None)
    assert(valid_pos != None)
    x_var = None
    if dataset_name == "swda":
        load_data = data_process.load_utterance_dataset(train_pos, valid_pos)
        # need dataset description
        # TODO: other dataset
        x_var = T.dtensor3('x_var')
    elif dataset_name == "microblog":
        logging.info("loading microblog data now!")
        load_data = data_process.load_microblogdata(train_pos, valid_pos, test_pos)
        x_var = T.dmatrix('x_var')


    """
    ========= Choose Model ==============
    """

    # define model
    assert(model_name != None)
    if model_name == "rcnn":
        assert(word_dim != None)
        assert(cnn_n_feature_maps != None)
        assert(cnn_window_sizes != None)
        assert(rnn_n_hidden != None)
        assert(rnn_n_output != None)

        run_model = rcnn.RCNN(rng=np.random.RandomState(54321),
                           input_data=x_var,
                           dim=word_dim,
                           n_feature_maps=cnn_n_feature_maps,
                           window_sizes=cnn_window_sizes,
                           n_hidden=rnn_n_hidden,
                           n_out=rnn_n_output)

        logging.info("model description:")
        logging.info("==================")
        logging.info("model type: rcnn")
        logging.info("n_feature_maps: %d" %(cnn_n_feature_maps))
        logging.info("window_sizes: {}".format(cnn_window_sizes))
        logging.info("n_hidden: %d" % (rnn_n_hidden))
        logging.info("n_out: %d" % (rnn_n_output))

    # TODO: other models!
    elif model_name == "rcnn_onestep_model":
        logging.info("define rcnn_onestep model now")
        assert(word_dim != None)
        assert(cnn_n_feature_maps != None)
        assert(cnn_window_sizes != None)
        assert(rnn_n_hidden != None)
        assert(rnn_n_output != None)

        h_pre_var = T.dvector('h_pre_var')
        run_model = models.RCNN_OneStep(x_var,
                                        word_dim,
                                        cnn_n_feature_maps,
                                        cnn_window_sizes,
                                        rnn_n_hidden,
                                        rnn_n_output,
                                        h_pre_var)
        logging.info("model description:")
        logging.info("==================")
        logging.info("model type: rcnn_onestep model")
        logging.info("n_feature_maps: %d" %(cnn_n_feature_maps))
        logging.info("window_sizes: {}".format(cnn_window_sizes))
        logging.info("n_hidden: %d" % (rnn_n_hidden))
        logging.info("n_out: %d" % (rnn_n_output))

    elif model_name == "srnn_trnn_model":
        logging.info("define srnn-trnn model now")
        assert(word_dim != None)
        assert(level1_input != None)
        assert(level1_hidden != None)
        assert(level1_output != None)
        assert(level2_input != None)
        assert(level2_hidden != None)
        assert(level2_output != None)
        logging.info("model description:")
        logging.info("=====================")
        logging.info("model type: srnn-trnn model")
        logging.info("the first level input layer num: %d"%(level1_input))
        logging.info("the first level hidden layer num: %d"%(level1_hidden))
        logging.info("the first level output layer num: %d"%(level1_output))
        logging.info("the second level input layer num: %d"%(level2_input))
        logging.info("the second level hidden layer num: %d"%(level2_hidden))
        logging.info("the second level output layer num: %d"%(level2_output))
        max_sen_len = 100
        run_model = models.SRNN_TRNN(x_var,
                                     level1_input,
                                     level1_hidden,
                                     level1_output,
                                     level2_input,
                                     level2_hidden,
                                     level2_output)
        """================ end srnn-trnn model  ===========  """
    # begin to experiment
    assert(run_model != None)
    assert(load_data != None)
    assert(batch_type != None)
    assert(batch_size != None)
    assert(valid_frequency != None)
    if experiment == "swda":
        run_swda_experiment(load_data,
                            run_model,
                            batch_type,
                            batch_size,
                            "sgd")
    elif experiment == "microblog":
        logging.info("begin to microblog experiment")
        run_microblog_experimentV2(load_data,
                                 run_model,
                                 model_name,
                                 batch_type,
                                 batch_size,
                                 n_epochs,
                                 valid_frequency,
                                 learning_rate,
                                 "sgd")
    # different dataset has different variable types
    # dtensor3, imatrix
