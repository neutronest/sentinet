## -*- coding: utf-8 -*-
import pdb
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import utils
import config

class CNN(object):

    def __init__(self,
                 input_var,
                 lookup_table,
                 dim,
                 n_feature_maps,
                 window_sizes,
                 n_output):
        """
        Params:
        -------
        input_data: symbolic sentence tensor, a 3-dimension tensor
        type of input_data: theano.tensor 3d tensor

        rng

        dim: the dimensions of word vector
        type of dim: int

        n_feature_maps: number of feature maps
        type of n_feature_maps: int

        window_size: the filters
        type of window_size: tuple of int


        """
        self.input_var = input_var
        self.window_sizes = window_sizes
        #self.input_data = input_data.dimshuffle('x', 'x', 0, 1)
        self.dim = dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes
        self.lookup_table =  utils.sharedX(lookup_table,
                                           dtype=theano.config.floatX,
                                           name='W_emb')
        self.n_output = n_output
        # params init
        self.params = []
        self.W_list = []
        rng = np.random.RandomState(23455)
        for ws in window_sizes:
            # ws declare each window_size
            W_init = np.asarray(rng.uniform(low=-0.1,
                                            high=0.1,
                                            size=(self.n_feature_maps,
                                                  1,
                                                  ws,
                                                  self.dim)),
                                dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name="cnn_"+str(ws))
            self.W_list.append(W)
            self.params.append(W)
        b_init = np.asarray(np.zeros((self.n_feature_maps * len(self.window_sizes),), dtype=theano.config.floatX))
        self.b = theano.shared(value=b_init)
        self.params.append(self.b)
        self.W_output = utils.shared_uniform((self.n_feature_maps*len(self.window_sizes), self.n_output),
                                             dtype=theano.config.floatX,
                                             name='cnn_W_output')
        self.by = utils.shared_zeros((self.n_output,),
                                     dtype=theano.config.floatX,
                                     name="cnn_by")
        self.params += [self.W_output, self.by, self.lookup_table]
        return

    def _conv(self, word_vector):
        #word_vector = utils.get_var_with_mask(word_vector, 0.)
        word_matrix = word_vector.dimshuffle('x', 'x', 0, 1)
        h = None

        for i in xrange(len(self.window_sizes)):
            conv_out = conv.conv2d(input=word_matrix, filters=self.W_list[i])
            max_out = T.max(conv_out, axis=2).flatten()
            h = max_out if h == None else \
                T.concatenate([h, max_out])
        o = h + self.b
        y = T.dot(o, self.W_output) + self.by
        return y

    def build_network(self):
        x_emb = self.lookup_table[self.input_var.flatten()].reshape([self.input_var.shape[0],
                                                                     self.input_var.shape[1],
                                                                     config.options['word_dim']])
        self.y, _ = theano.scan(fn=self._conv,
                                sequences=x_emb,
                                n_steps=self.input_var.shape[0],
                                outputs_info=None)
        #pdb.set_trace()
        return

class SCNN(object):
    """ the CNN that apply sequence data """
    def __init__(self,
                 input_var,
                 word_dim,
                 n_feature_maps,
                 window_sizes):
        """

        """
        self.input_var = input_var
        self.word_dim = word_dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes
        # params init
        self.params = []
        self.W_list = []
        self.sens_pos_var = T.imatrix('sens_pos_var')
        for ws in window_sizes:
            # ws declare each window_size
            W = utils.shared_uniform((self.n_feature_maps,
                                      1,
                                      ws,
                                      self.word_dim),
                                     dtype=theano.config.floatX,
                                     name="cnn_"+str(ws))
            self.W_list.append(W)
            self.params.append(W)

        self.b = utils.shared_zeros((self.n_feature_maps*len(self.window_sizes),),
                                    dtype=theano.config.floatX)
        self.params.append(self.b)
        return

    def _conv(self, sen_pos):

        word_matrix = self.input_var[sen_pos[0]:sen_pos[1]]
        #word_matrix = utils.get_var_with_mask(word_matrix, 0.)
        word_tensor = word_matrix.dimshuffle('x', 'x', 0, 1)
        h = None

        for i in xrange(len(self.window_sizes)):
            conv_out = conv.conv2d(input=word_tensor, filters=self.W_list[i])
            max_out = T.max(conv_out, axis=2).flatten()
            h = max_out if h == None else \
                T.concatenate([h, max_out])
        o = h + self.b
        return o

    def build_network(self):
        output, _ = theano.scan(fn=self._conv,
                                sequences=self.sens_pos_var,
                                outputs_info=None)
        self.h = output
        return

class CNN_OneStep(object):
    """ the cnn with batch size=1
    """

    def __init__(self,
                 input_var,
                 dim,
                 n_feature_maps,
                 window_sizes):

        self.input_var = input_var
        self.dim = dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes

        self.params = []
        self.cnn_weights = []
        for ws in window_sizes:
            cnn_weight = utils.shared_uniform((n_feature_maps, 1, ws, self.dim),
                                              dtype=theano.config.floatX, name="cnn"+str(ws))

            self.params.append(cnn_weight)
            self.cnn_weights.append(cnn_weight)

        self.b = utils.shared_zeros((self.n_feature_maps* len(self.window_sizes)), dtype=theano.config.floatX, name='b')
        self.params.append(self.b)
        return

    def build_network(self):
        word_matrix = self.input_var.dimshuffle('x', 'x', 0, 1)
        h = None

        for i in xrange(len(self.window_sizes)):
            conv_out = conv.conv2d(input=word_matrix,
                                   filters=self.cnn_weights[i])
            max_out = T.max(conv_out, axis=2).flatten()

            if h == None:
                h = max_out
            else:
                h = T.concatenate([h, max_out])

        # the dimension of h is len(window_sizes) * n_feature_maps
        self.output = h + self.b
        return



def test_cnn():

    x_var = T.dtensor3('x_var')
    rng = np.random.RandomState(54321)
    n_feature_maps = 300
    window_sizes = (2, 3, 4)
    dim = 300
    cnn = CNN(x_var, rng, dim, n_feature_maps, window_sizes)

    cnn_output_fn = theano.function(inputs=[x_var], outputs=cnn.output)

    # define each sentence matrix with fixed word vector and fixed sentence length
    seq_len = 5
    sen_len = 10
    seq = []
    for i in xrange(seq_len):
        sen_emb = np.asarray(np.random.uniform(size=(sen_len, dim),
                             low=-.01, high=.01),
                             dtype=theano.config.floatX)
        seq.append(sen_emb)
    seq_x = np.asarray(seq, dtype=theano.config.floatX)
    output = cnn_output_fn(seq_x)
    assert(output.shape[0] == seq_len)
    assert(output.shape[1] == n_feature_maps*len(window_sizes))
    print "[CNN Test OK!]"
    return cnn

if __name__ == "__main__":
    test_cnn()
