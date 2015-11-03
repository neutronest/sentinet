# -*- coding: utf-8 -*-
import pdb
import theano, theano.tensor as T
import rnn
import cnn
import utils


class RNN_TRNN(object):
    """
    the rnn-(tree rnn) model

    first using RNN to generate sentence vector,
    then appling the sentence vector to the tree rnn
    """
    def __init__(self,
                 input_var,
                 word_dim,
                 sens_pos,
                 relation_pairs,
                 rnn_hidden,
                 rnn_output,
                 trnn_input,
                 trnn_hidden,
                 trnn_output):
        """
        the RNN-TRNN model

        input_var: the input variable
            type: theano tensor of dvector

        word_dim: the dimension of word vectors
            type: int (default 300)

        sens_pos: the ndarray store the start pos / end pos
                        of each sentence in the word flatten vector
            type: theano tensor of imatrix

        relation_pairs: the

        """

        self.input_var = input_var
        self.word_dim = word_dim
        self.sens_pos = sens_pos
        self.relation_pairs = relation_pairs
        self.rnn_hidden = rnn_hidden
        self.rnn_output = rnn_output # abandon
        self.trnn_input = trnn_input
        self.trnn_hidden = trnn_hidden
        self.trnn_output = trnn_output

        self.rnn_model = rnn.RNN(self.word_dim,
                            self.rnn_hidden,
                            self.rnn_output)
        self.trnn_model = rnn.TRNN(self.trnn_input,
                                   self.trnn_hidden,
                                   self.trnn_output)

        return

    def _get_rnn_hidden_state(self, sen_pos, sens_var):
        words_var = sens_var.take([i for i in xrange(sen_pos[0], sen_pos[1])])
        return self.rnn_model.build_network(words_var)

    def build_network(self):

        [y_pred_var, h_var, output, loss, error], _ = theano.scan(fn=self._get_rnn_hidden_state,
                                                                  sequences=[self.sens_pos],
                                                                  non_sequences=[self.input_var],
                                                                  outputs_info=None)
        # we get a list of h_states with order



        # first level RNN


        return





class RCNN_OneStep(object):
    """ The RCNN model with batch 1

    """

    def __init__(self,
                 input_var,
                 y_pre_var,
                 word_dim,
                 cnn_feature_maps,
                 cnn_window_sizes,
                 rnn_hidden,
                 rnn_output,
                 h_tm1):

        self.cnn_onestep_model = cnn.CNN_OneStep(input_var,
                                           word_dim,
                                           cnn_feature_maps,
                                           cnn_window_sizes)
        self.cnn_onestep_model.build_network()
        self.rnn_onestep_model = rnn.RNN_OneStep(self.cnn_onestep_model.output,
                                                 y_pre_var,
                                                 cnn_feature_maps*len(cnn_window_sizes),
                                                 rnn_hidden,
                                                 rnn_output,
                                                 h_tm1)
        self.params = self.cnn_onestep_model.params + self.rnn_onestep_model.params
        self.input_var = input_var
        self.y_pre_var = y_pre_var
        self.h_pre_var = h_tm1
        self.word_dim = word_dim
        self.rnn_hidden = rnn_hidden
        self.rnn_onestep_model.build_network()
        self.h = self.rnn_onestep_model.h
        self.loss = self.rnn_onestep_model.loss
        self.error = self.rnn_onestep_model.error
        self.y_pred = self.rnn_onestep_model.y_pred
        self.output_var = self.rnn_onestep_model.output_var
        return
