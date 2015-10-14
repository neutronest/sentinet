# -*- coding; utf-8 -*-
import pdb
import theano, theano.tensor as T
import numpy as np
import logging
import random
import optimizer
import data_process
import utils
import sys
import rcnn

def start_rnn_with_cnn(dim,
                       n_feature_maps,
                       window_sizes,
                       n_hidden,
                       n_out,
                       learning_rate=0.01,
                       learning_rate_decay=0.99,
                       n_epochs=50,
                       batch_size=10,
                       train_pos=1000,
                       valid_pos=1010):

    """
    Parameters:
    -----------
    dim: the dimension of word feature
        type: int

    n_feature_maps: the num of feature maps
        type: int

    window_sizes: the filter sizes
        type: list of int

    n_hidden: the num of hidden units in rnn
        type: int

    n_out: the num of output units in rnn
        type: int

    learning_rate: just the learning rate of SGD
        type: float

    n_epochs: the num of epochs when trained
        type: int

    batch_size: if use mini batch, choose the
                train data length in each batch
        type: int

    train_pos: The position of train data ending in the whole dataset
        type: int

    valid_pos: The position of valid data ending in the whole dataset
        type: int

    HINT: the train dataset:  data[:train_pos]
          the valid dataset:  data[train_pos:valid_pos]
          the test dataset: data[valid_pos:]
    """

    # log the params
    logging.info("start the rcnn process!")
    logging.info("the params of this turn is:")
    logging.info("n_feature_maps: %d" % (n_feature_maps))
    logging.info("window_sizes: {}".format(window_sizes))
    logging.info("n_hidden: %d" % (n_hidden))
    logging.info("learning_rate: %f" % (learning_rate))
    logging.info("initialize")

    #x_var_iter = T.dtensor3('x_var_iter')
    #y_var_iter = T.imatrix('y_var_iter')
    x_var = T.dtensor3('x_var')
    y_var = T.imatrix('y_var')
    lr_var = T.scalar('lr_var')
    label_var = T.vector('label_var')


    rcnn_model = rcnn.RCNN(rng=np.random.RandomState(54321),
                           input_data=x_var,
                           dim=dim,
                           n_feature_maps=n_feature_maps,
                           window_sizes=window_sizes,
                           n_hidden=n_hidden,
                           n_out=n_out)
    cost = rcnn_model.loss(rcnn_model.y, y_var)
    #gparams_var = [T.grad(cost, param) for param in rcnn.params]
    gparams = [T.grad(cost, param_var) for param_var in rcnn_model.params]
    sgd_updates = {}
    for param, gparam in zip(rcnn_model.params, gparams):
        ugd = - gparam * lr_var
        sgd_updates[param] = param + ugd


    #compute_gradients = theano.function(inputs=[x_var, y_var],
    #                                    outputs=gparams)
    train_loss = theano.function(inputs=[x_var, y_var, lr_var],
                                 outputs=[cost],
                                 updates=sgd_updates)
    """
    sgd_updates = {}
    gparams = [T.grad(cost, param) for param in rcnn.params]
    for param, gparam in zip(rcnn.params, gparams):
        ugd = - learning_rate * gparam
        sgd_updates[param] = param + ugd
    """

    # definite loss function and SGD funcition
    logging.info("begin to train")
    train_x, train_y, valid_x, valid_y, test_x, test_y = data_process.load_utterance_dataset(train_pos, valid_pos)
    # get dataset length
    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    epoch = 0

    """
    train_model_fn = theano.function(inputs=[x_var, y_var, lr_var],
                                     updates=sgd_updates,
                                     outputs=[])
    """
    while epoch < n_epochs:
        epoch += 1
        #gparams_acc = None
        train_losses = 0.
        # mini batch
        logging.info("prepare the %i's mini batch"%(epoch))
        logging.info("the size of batch is %i"%(batch_size))
        batch_index = random.randint(0, 99)
        batch_start = batch_index * batch_size
        batch_stop = batch_start + batch_size
        for idx in xrange(batch_start, batch_stop):
            # accumulate gradients
            train_loss = train_loss(utils.wrap_x(train_x[idx]),
                                    utils.expand_y(train_x[idx], 43),
                                    learning_rate)
            train_losses += train_loss
            logging.info("the seq %i's train loss is: %f"%(train_loss))
            logging.info("epoch %i's train losses is: %f" %(epoch, train_losses))
        # update the params

if __name__ == "__main__":

    log_file = sys.argv[1]
    logging.basicConfig(
        level=logging.DEBUG, format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', filename=log_file, filemode='w')
    start_rnn_with_cnn(dim=300,
                       n_feature_maps=300,
                       window_sizes=(2, 3, 4, 5),
                       n_hidden=300,
                       n_out=43,
                       learning_rate=0.01,
                       learning_rate_decay=0.99,
                       batch_size=10,
                       n_epochs=50,
                       train_pos=1000,
                       valid_pos=1006)
