# -*- coding: utf-8 -*-
import sys, getopt
import pdb
import rcnn
import optimizer
import data_process
import logging
import theano, theano.tensor as T
import numpy as np
import random
from collections import OrderedDict
import utils


def run_swda_experiment(load_data, 
                        model, 
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
    cost = None
    error = None
    # optimizer define
    gparams = None
    optimizer_updates = OrderedDict()

    # standard batchsize or mini batch
    if batch_size == 10:
        y_var = T.imatrix('y_var')
        lr_var = T.scalar('lr_var')
        label_var = T.vector('label_var')
        cost = model.loss(model.y, y_var)
        error = model.error(label_var, model.output_var)
    else:
        y_var = [T.imatrix()] * batch_size
        label_var = [T.vector()] * batch_size
        cost = None
        for i in xrange(0, batch_size):
            cost += model.loss(model.y, y_var[i])
        lr_var = T.scalar("lr_var")

    
    if optimizer == "sgd":
        gparams = [T.grad(cost, param_var) for param_var in model.params]
        optimizer_updates = [(param, param - gparam * lr_var) \
            for param, gparam in zip(model.params, gparams)]
        """
        for param, gparam in zip(model.params, gparams):
            ugd = - gparam * lr_var
            optimizer_updates[param] = param + ugd
        """

    #compute_gradients = theano.function(inputs=[x_var, y_var],
    #                                    outputs=gparams)
    train_loss_fn = theano.function(inputs=[x_var, y_var, lr_var],
                                    outputs=[cost],
                                    updates=optimizer_updates)

    compute_loss_fn = theano.function(inputs=[x_var, y_var],
                                      outputs=[cost],
                                      mode="DebugMode")
    compute_error_fn = theano.function(inputs=[x_var, label_var],
                                       outputs=[error],
                                       mode="DebugMode")

    print "begin to train"
    logging.info("begin to train")
    while epoch < n_epochs:
        epoch += 1
        #gparams_acc = None
        train_losses = 0.
        # mini batch
        logging.info("prepare the %i's mini batch"%(epoch))
        logging.info("the size of batch is %i"%(batch_size))

        batch_index_list = [x for x in xrange(0, 1000, 10)]

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

def run_microblog_experiment():
    """
    the rnn experiment with microblog dataset
    """

    return


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
    dropout_rate = None
    optimizer_method = None
    learning_rate = None
    batch_size = None
    n_epochs = None
    train_pos = None
    valid_pos = None

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
        "rnn_n_out=",
        "dropout_rate=", 
        "optimizer_method=", 
        "learning_rate=",
        "batch_size=",
        "n_epochs=",
        "train_pos=",
        "valid_pos="])
    
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

        elif opt == "--rnn_n_out":
            # example of arg: 43 of str
            rnn_n_out = int(arg)

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

        elif opt == "--batch_size":
            # example of arg: 10 of str
            batch_size = int(arg)

        elif opt == "--n_epochs":
            # example of arg: 100 of str
            n_epochs = int(arg)

        elif opt == "--train_pos":
            # example of arg: 1000 of str
            train_pos = int(arg)

        elif opt == "--valid_pos":
            # example of arg: 1005 of str
            valid_pos = int(arg)

    # define log file
    print "prepare the logging file"
    assert(log_path != None)  
    logging.basicConfig(
        level=logging.DEBUG, format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', filename=log_path, filemode='w')


    # define dataset
    assert(dataset_name != None)
    assert(train_pos != None)
    assert(valid_pos != None)
    if dataset_name == "swda":
        load_data = data_process.load_utterance_dataset(train_pos, valid_pos)
        # need dataset description
        # TODO: other dataset

    # define model
    assert(model_name != None)
    x_var = T.dtensor3('x_var')

    if model_name == "rcnn":
        assert(word_dim != None)
        assert(cnn_n_feature_maps != None)
        assert(cnn_window_sizes != None)
        assert(rnn_n_hidden != None)
        assert(rnn_n_out != None)

        run_model = rcnn.RCNN(rng=np.random.RandomState(54321),
                           dim=word_dim,
                           input_data=x_var,
                           n_feature_maps=cnn_n_feature_maps,
                           window_sizes=cnn_window_sizes,
                           n_hidden=rnn_n_hidden,
                           n_out=rnn_n_out)

        logging.info("model description:")
        logging.info("==================")
        logging.info("model type: rcnn")
        logging.info("n_feature_maps: %d" %(cnn_n_feature_maps))
        logging.info("window_sizes: {}".format(cnn_window_sizes))
        logging.info("n_hidden: %d" % (rnn_n_hidden))
        logging.info("n_out: %d" % (rnn_n_out))
    
    # TODO: other models!
    else:
        pass

    # begin to experiment
    assert(run_model != None)
    assert(load_data != None)
    if experiment == "swda":
        run_swda_experiment(load_data,
                            run_model,
                            batch_size,
                            "sgd")

    # different dataset has different variable types
    # dtensor3, imatrix


