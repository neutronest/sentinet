# -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import sys

sys.path.append("../src")
import rnn
import cnn
import models
import utils
import data_process

#theano.config.on_unused_input='warn'
#theano.config.exception_verbosity='high'
theano.config.optimizer='fast_compile'

def test_rcnn_onestep():
    """
    """
    # prepare dataset

    sen_len = 10
    word_dim = 300
    # prepare data
    sen = []
    for i in xrange(sen_len):
        word_emb = utils.ndarray_uniform((word_dim,), dtype=theano.config.floatX)
        sen.append(word_emb)

    sen_x = np.asarray(sen, dtype=theano.config.floatX)
    h_0 = utils.ndarray_uniform((300,), 0.05)
    y_true = np.asarray([0, 1, 0], dtype=np.int32)

    # param
    input_var = T.dmatrix('input_var')
    y_var = T.ivector('y_var')
    label_var = T.scalar('label_var')
    cnn_feature_maps = 300
    cnn_window_sizes = (2,3,4)
    rnn_hidden = 300
    rnn_output = 3
    h_tm1 = T.dvector('h_tm1')

    rcnn_onestep_model = models.RCNN_OneStep(input_var,
                                       word_dim,
                                       cnn_feature_maps,
                                       cnn_window_sizes,
                                       rnn_hidden,
                                       rnn_output,
                                       h_tm1)
    cost_var = rcnn_onestep_model.loss(y_var, rcnn_onestep_model.y_pred)
    error_var = rcnn_onestep_model.error(label_var, rcnn_onestep_model.output_var)
    compute_cost_fn = theano.function(inputs=[input_var, y_var, h_tm1],
                                      outputs=[cost_var, rcnn_onestep_model.h])
    compute_error_fn = theano.function(inputs=[input_var, label_var, h_tm1],
                                       outputs=[error_var, rcnn_onestep_model.output_var])

    [cost, h_current] = compute_cost_fn(sen_x, y_true, h_0)
    [error, label_pred] = compute_error_fn(sen_x, 1, h_0)
    print "the cost of rcnn_model_onestep is %f"%(cost)
    print "the error of rcnn_model_onestep is %d"%(error)
    print "the predict label of rcnn_model_onestep is %d"%(label_pred)
    print "the current h of rcnn_model_onestep is :"
    print h_current
    print "[Test rcnn_model_onestep OK!]"
    print "====="
    return

def test_srnn_trnn():
    """
    """
    srnn_input = 5
    srnn_hidden = 100
    srnn_output = 3
    trnn_input = 100
    trnn_hidden = 500
    trnn_output = 3

    input_var = T.dmatrix('input_var')
    sens_pos_ndarr = np.asarray([[0, 2], [3,7], [8,10]], dtype=np.int32)
    sens_pos = theano.shared(sens_pos_ndarr)
    sens = utils.ndarray_uniform((10, 10),
                                 dtype=theano.config.floatX)
    relations = np.asarray([[0, -1],
                            [1, 0],
                            [2, 0]], dtype=np.int32)
    tree_states = np.asarray(np.zeros((4, trnn_hidden),
                                      dtype=theano.config.floatX))
    max_sens_size = 100
    srnn_trnn_model = models.SRNN_TRNN(input_var,
                                       srnn_input,
                                       srnn_hidden,
                                       srnn_output,
                                       trnn_input,
                                       trnn_hidden,
                                       trnn_output)
    get_trnn_h_fn = theano.function(inputs=[input_var,
                                            srnn_trnn_model.sens_pos_var,
                                            srnn_trnn_model.relation_pairs,
                                            srnn_trnn_model.trnn_model.th0],
                                    outputs=[srnn_trnn_model.y_pred])

    get_srnn_h_fn = theano.function(inputs=[input_var,
                                            srnn_trnn_model.sens_pos_var],
                                    outputs=srnn_trnn_model.srnn_model.hidden_states_var)
    #pdb.set_trace()
    #(train_x, train_y, valid_x, valid_y, test_x, test_y) = data_process.load_microblogdata((0, 1, 2), 3, 4)

    #x = train_x['3512534461952614'][0]
    #sens_pos = train_x['3512534461952614'][1]
    #relations = train_x['3512534461952614'][2]
    x = utils.ndarray_uniform((1,5),
                              dtype=theano.config.floatX)
    sens_pos=[[0,1]]
    relations= [[0,-1]]
    th_first = np.zeros((1,trnn_hidden),
                        dtype=theano.config.floatX)
    #y = get_trnn_h_fn(x, sens_pos, relations)
    hs = get_srnn_h_fn(x, sens_pos)
    #print hs
    y = get_trnn_h_fn(x, sens_pos, relations, th_first)
    print "[Test SRNN-TRNN OK!]"
    return

def test_sgru_tgru():
    """
    """
    n_input = 10
    n_hidden = 50
    n_output = 3
    input_var = T.dmatrix('input_var')
    sens_pos_ndarr = np.asarray([[0, 2], [3,7], [8,10]], dtype=np.int32)
    sens_pos = theano.shared(sens_pos_ndarr)
    sens = utils.ndarray_uniform((10, 10),
                                 dtype=theano.config.floatX)
    relations = np.asarray([[0, -1],
                            [1, 0],
                            [2, 0]], dtype=np.int32)

    tree = np.asarray(np.zeros((n_hidden*(len(relations)+1),),
                               dtype=theano.config.floatX))



    sgru_tgru_model = models.SGRU_TGRU(input_var,
                                       n_input,
                                       n_hidden,
                                       n_output,
                                       n_hidden,
                                       n_hidden,
                                       n_output)

    get_model_y_fn = theano.function(inputs=[input_var,
                                             sgru_tgru_model.sens_pos_var,
                                             sgru_tgru_model
                                             .relation_pairs,
                                             sgru_tgru_model.th],
                                    outputs=[sgru_tgru_model.y_pred])

    y_res = get_model_y_fn(sens,
                           sens_pos_ndarr,
                           relations,
                           tree)
    print y_res

    return

def test_slstm_tlstm():
    """

    """
    level1_input = 10
    level1_hidden = 50
    level1_output = 3
    level2_input = 50
    level2_hidden = 50
    level2_output = 3
    input_var = T.dmatrix('input_var')
    sens_pos_ndarr = np.asarray([[0, 2], [3,7], [8,10]], dtype=np.int32)
    sens_pos = theano.shared(sens_pos_ndarr)
    sens = utils.ndarray_uniform((10, 10),
                                 dtype=theano.config.floatX)
    relations = np.asarray([[0, -1],
                            [1, 0],
                            [2, 0]], dtype=np.int32)

    th0 = np.asarray(np.zeros((level2_hidden*(len(relations)+1),),
                               dtype=theano.config.floatX))
    tc0 = np.asarray(np.zeros((level2_hidden*(len(relations)+1),),
                               dtype=theano.config.floatX))

    slstm_tlstm_model = models.SLSTM_TLSTM(input_var,
                                           level1_input,
                                           level1_hidden,
                                           level1_output,
                                           level2_input,
                                           level2_hidden,
                                           level2_output)

    get_model_y_fn = theano.function(inputs=[input_var,
                                             slstm_tlstm_model.sens_pos_var,
                                             slstm_tlstm_model.relation_pairs,
                                             slstm_tlstm_model.th,
                                             slstm_tlstm_model.tc,],
                                     outputs=[slstm_tlstm_model.y_pred])

    y_res = get_model_y_fn(sens,
                           sens_pos_ndarr,
                           relations,
                           th0,
                           tc0)
    print y_res
    return


def test_scnn_trnn():
    level1_input = 10
    level1_hidden = 50
    level2_input = 50
    level2_hidden = 50
    n_output = 3
    word_dim = 200
    n_feature_maps = 100
    window_sizes  = (2,3)
    input_var = T.dmatrix('input_var')

    scnn_trnn_model = models.SCNN_TRNN(input_var,
                                       word_dim,
                                       n_feature_maps,
                                       window_sizes,
                                       level2_input,
                                       level2_hidden,
                                       n_output
    )


def test_model():
    level1_model_names = ['scnn_model',
                          'srnn_model',
                          'sgru_model',
                          'slstm_model']
    level2_model_names = ['trnn_model',
                          'tgru_model',
                          'tlstm_model']
    input_var = T.dmatrix('input_var')
    level1_input = 10
    level1_hidden = 50
    level2_input = 50
    level2_hidden = 50
    n_output = 3
    word_dim = 10
    n_feature_maps = 100
    window_sizes = [1,2,3]

    sens_pos_ndarr = np.asarray([[0, 2], [3,5], [6,9]], dtype=np.int32)
    sens_pos = theano.shared(sens_pos_ndarr)
    sens = utils.ndarray_uniform((10, 10),
                                 dtype=theano.config.floatX)
    relations = np.asarray([[0, -1],
                            [1, 0],
                            [2, 0]], dtype=np.int32)

    th0 = np.asarray(np.zeros((level2_hidden*(len(relations)+1),),
                               dtype=theano.config.floatX))
    tc0 = np.asarray(np.zeros((level2_hidden*(len(relations)+1),),
                               dtype=theano.config.floatX))



    for model1_name in level1_model_names:
        for model2_name in level2_model_names:

            # trick
            #if model1_name == "scnn_model":
            #    continue

            print "%s-%s model test!"%(model1_name, model2_name)
            run_model = models.Model(model1_name,
                                     model2_name,
                                     input_var,
                                     level1_input,
                                     level1_hidden,
                                     level2_input,
                                     level2_hidden,
                                     n_output,
                                     word_dim,
                                     n_feature_maps,
                                     window_sizes)
            print "[Build model OK!]"

            if model2_name == "tlstm_model":
                get_model_y_fn = theano.function(inputs=[input_var,
                                             run_model.sens_pos_var,
                                             run_model.relation_pairs,
                                             run_model.th,
                                             run_model.tc,],
                                     outputs=[run_model.y_drop_pred])
                y_res = get_model_y_fn(sens,
                                       sens_pos_ndarr,
                                       relations,
                                       th0,
                                       tc0)
                print y_res
            else:
                get_model_y_fn = theano.function(inputs=[input_var,
                                             run_model.sens_pos_var,
                                             run_model.relation_pairs,
                                             run_model.th],
                                                 outputs=[run_model.y_drop_pred])
                y_res = get_model_y_fn(sens,
                                       sens_pos_ndarr,
                                       relations,
                                       th0)
                print y_res
    return


if __name__ == "__main__":
    test_model()
