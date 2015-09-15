# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np


from vectorize import Vectorize
import data_process
import rnn
import cnn
import pdb
import logging
import sys


class RCNN(object):

    def __init__(self,
                 rng,
                 input_data,
                 dim,
                 n_feature_maps,
                 window_sizes,
                 n_hidden,
                 n_out
                ):
        self.cnn = cnn.CNN(input_data=input_data,
                           rng=rng,
                           dim=dim,
                           n_feature_maps=n_feature_maps,
                           window_sizes=window_sizes)
        self.rnn = rnn.RNN(input_data=self.cnn.output,
                           rng=rng,
                           n_input = n_feature_maps*len(window_sizes),
                           n_hidden=n_hidden,
                           n_output=n_out,
                           activation=T.nnet.sigmoid,
                           output_type="softmax")

        self.window_sizes = window_sizes
        self.dim = dim
        self.n_out = n_out
        self.n_hidden = self.rnn.n_hidden
        self.params = self.cnn.params + self.rnn.params
        self.output = self.rnn.output_var
        self.loss = self.rnn.loss
        self.error = self.rnn.error
        return


def start_rcnn(dim,
               n_feature_maps,
               window_sizes,
               n_hidden,
               n_out,
               learning_rate=0.01,
               learning_rate_decay=0.1,
               n_epochs=1,
               validation_frequency=1000):


    # log the params
    logging.info("start the rcnn process!")
    logging.info("the params of this turn is:")
    logging.info("n_feature_maps: %d" % (n_feature_maps))
    logging.info("window_sizes: {}".format(window_sizes))
    logging.info("n_hidden: %d" % (n_hidden))
    logging.info("learning_rate: %f, learning_rate_decay: %f, validation_frequency: %d" % (learning_rate, learning_rate_decay, validation_frequency))

    logging.info("initialize")
    x_var = T.dtensor3('x_var')
    y_var = T.ivector('y_var')
    lr_var = T.scalar("lr_var")
    label_var = T.vector('label_var')
    rcnn = RCNN(rng=np.random.RandomState(54321),
                input_data=x_var,
                dim=dim,
                n_feature_maps=n_feature_maps,
                window_sizes=window_sizes,
                n_hidden=n_hidden,
                n_out=n_out)

    # definite loss function and SGD funcition
    cost = rcnn.loss(y_var)
    error = rcnn.error(label_var)
    gparams = [T.grad(cost, param) for param in rcnn.params]
    sgd_updates = [(param, param - lr_var * gparam) for (param, gparam) in zip(rcnn.params, gparams)]

    # definite train model
    train_model = theano.function(inputs=[x_var, y_var, lr_var],
                                  outputs=[],
                                  updates=sgd_updates)
    compute_loss = theano.function(inputs=[x_var, y_var],
                                   outputs=[cost])
    compute_error = theano.function(inputs=[x_var, label_var],
                                    outputs=[error],
                                    givens={

                                    })

    logging.info("begin to train")
    # load the dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y = data_process.load_utterance_dataset()

    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    epoch = 0

    while epoch < n_epochs:
        epoch += 1
        for idx in xrange(n_train):
            train_model(train_x[idx],
                        train_y[idx],
                        learning_rate)

            valid_iter = (epoch-1) * n_train + idx + 1
            if valid_iter % validation_frequency == 0:

                error_cnt = 0
                cost_cnt = 0
                for vdx in xrange(n_valid):
                    valid_cost = compute_loss(valid_x[vdx], valid_y[vdx])
                    valid_error = compute_error(valid_x[vdx], valid_y[vdx])
                    cost_cnt += valid_cost
                    error_cnt += valid_error

                # valid cost and error stats
                error_cnt = error_cnt * 1.0 / n_valid
                cost_cnt = cost_cnt * 1.0 / n_valid
                logging.info("Epoch %d, seq %i/%i, error_cnt: %f, cost_cnt: %f" \
                             %(epoch, idx, n_train, error_cnt, cost_cnt))

        learning_rate = learning_rate * (1-learning_rate_decay)

if __name__ == "__main__":

    log_file = sys.argv[1]
    logging.basicConfig(
        level=logging.DEBUG, format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', filename=log_file, filemode='w')
    start_rcnn(dim=300,
              n_feature_maps=300,
              window_sizes=(2, 3,4 , 5),
              n_hidden=300,
              n_out=43)
