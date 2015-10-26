# -*- coding: utf-8 -*-
import pdb
from collections import OrderedDict
from vectorize import Vectorize
import logging
import numpy as np
import theano
import json
import sys
sys.path.append("../microblog")
from mvectorize import MVectorize

"""
============================ SWDA utterances ===============================

All functions about utterances:

* gen_structured_data_from_utterances
* load_utterances_dataset

"""


def gen_structured_data_from_utterances(filename):
    """

    Parameters:
    -----------
    filename: the file location of utterances
              type: string

    Return:
    -------
    thead_data:
     the thread_data format:
    [
       # each thread

       {
         "sen_1": [label, sentence_words, sentence_vector],
         "sen_2": [..]
       },
       {
           ...
       }
    ]
    """
    thread_data = []
    sen_cur = 0
    y_dict = {}
    y_idx = 0
    sen = []
    sen_dict = OrderedDict()
    sen_tuple = None
    gap_cur = 0
    with open(filename, "r") as file_ob:
        for line in file_ob:
            line_arr = line.strip().split("\t")

            # the split mark of dataset
            if line_arr[0] == "==":
                # next dialog
                thread_data.append(sen_dict)
                # re init
                sen_dict = OrderedDict()
                continue
            elif len(line_arr) == 1:
                #print "mind the gap"
                #print gap_cur
                gap_cur += 1
                continue
            tag = line_arr[0]
            sen = line_arr[1]


            if y_dict.get(tag) == None:
                y_dict[tag] = y_idx
                y_idx += 1
            sen_tuple = [tag, sen, None]
            sen_sig = "sen_" + str(sen_cur)
            sen_dict[sen_sig] = sen_tuple
            sen_cur += 1
    file_ob.close()
    return thread_data

def load_utterance_dataset(train_pos, valid_pos):
    """
    """
    thread_data =  gen_structured_data_from_utterances("../data/utterances.txt")
    print "load_utterance_dataset"
    sens = train_sen_flatten(thread_data)

    # Vectorize Init...
    vectorize = Vectorize()
    logging.info("training the sentence model")
    vectorize.train_google_model("../data/G.bin")
    logging.info("the google model Done!")

    # apply the word vector to the thread data
    for sen_dict in thread_data:

        # get max sen len
        max_sen_len = 0
        len_of_sens = [len(v[1]) for k,v in sen_dict.items()]
        max_sen_len = max(len_of_sens)
        for k, v in sen_dict.items():
            sen_vec = np.zeros((max_sen_len, 300))
            for i in xrange(len(v[1])):
                try:
                    sen_vec[i] = vectorize.google_model[v[1][i]]
                except:
                    continue
            v[2] = sen_vec

    # gen the train_x, train_y
    """
    thread_x, thread_y = gen_structured_xy(thread_data)
    # pdb.set_trace()
    logging.info(" rnn start training")
    # prepare train_data and test_data
    data_x = [data for thread in thread_x for data in thread]
    data_y = [data for thread in thread_y for data in thread]

    TRAIN_SET = 200000
    VALID_SET = 210000
    print "data len: ", len(data_x)
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    valid_x = data_x[TRAIN_SET:VALID_SET]
    valid_y = data_y[TRAIN_SET:VALID_SET]
    test_x = data_x[VALID_SET:]
    test_y = data_y[VALID_SET:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y
    """
    data_x, data_y = gen_structured_xy(thread_data)
    TRAIN_SET = train_pos
    VALID_SET = valid_pos
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    valid_x = data_x[TRAIN_SET:VALID_SET]
    valid_y = data_y[TRAIN_SET:VALID_SET]
    test_x = data_x[VALID_SET:]
    test_y = data_y[VALID_SET:]
    return (train_x, train_y, valid_x, valid_y, test_x, test_y)

"""
================  END SWDA utterances ==================================================

================ BEGIN microblog data process ==========================================


"""

ERROR_FIND = 0
SUCCESS_FIND = 0

def generate_words_emb(words, mvectorize):
    """
    """
    global ERROR_FIND
    global SUCCESS_FIND
    words_emb = []
    for word in words:
        try:
            SUCCESS_FIND += 1
            word_vector = mvectorize.words_model[word]
            words_emb.append(word_vector)
        except:
            #print "[ERROR FIND]"
            ERROR_FIND += 1
            continue

    if len(words_emb) < 5:
        remain_dim = 5 - len(words_emb)
        # TODO: use 300 here is MAGIC! Need to touch
        for i in xrange(remain_dim):
            word_vector = np.asarray(np.zeros((300,), dtype=theano.config.floatX))
            words_emb.append(word_vector)
    return np.asarray(words_emb, dtype=theano.config.floatX)



def generate_threads(file_path, mvectorize, data_x, data_y):
    """ get the well-structured data of train/valid/test

    Parameters:
    -----------
    file_path: the sepcific file path
        type: str

    vectorize: the vectorize object
        type: vectorize object....

    data_x: the X of train/valid/test
        type: {threadid:[(docid, parentid, words_emb)]  }
               threadid: str
               docid: str
               parentid: str
               words_emb: matrix of ndarray

    data_y: the Y of train/valid/test
        type: {threadid: [(docid, label)]}
    """
    with open(file_path, "r") as train_ob:
        for line in train_ob:
            line = line.strip()
            line_json = json.loads(line)

            # parse the conponent
            threadid = str(line_json['threadid'])
            docid = str(line_json['docid'])
            parent = str(line_json['parent'])
            words = line_json['words']
            words_emb = generate_words_emb(words, mvectorize)

            label = int(line_json['label'])
            cur_item_x = (docid, parent, words_emb)
            cur_item_y = (docid, label)
            if data_x.get(threadid) == None:
                data_x[threadid] = [cur_item_x]
            else:
                data_x[threadid].append(cur_item_x)

            if data_y.get(threadid) == None:
                data_y[threadid] = [cur_item_y]
            else:
                data_y[threadid].append(cur_item_y)


    return (data_x, data_y)

def load_microblogdata(train_indicators,
                       valid_indicator,
                       test_indicator):
    """
    load the microblog tree-structure dataset

    Parameters:
    ----------

    train_indicators: the selected data dirs that used for train dataet
        type: tuple of int
        example: (1,2,3)

    valid_indicator: the selected data dir that used for valid dataset
        type: int

    tet_indicator:  the selected data dir that used for valid dataset
        type: int
    """

    # vectorzie init
    dir_path = "../data/weibo/fold_data/"
    mv = MVectorize()
    mv.gen_words_vector("../data/weibo/weiboV2.tsv")



    train_x = {}
    train_y = {}
    valid_x = {}
    valid_y = {}
    test_x = {}
    test_y = {}

    n_topics = 51
    print "=== generate train dataset"
    # generate train dataset
    for i in train_indicators:
        for topic_id in xrange(n_topics):
            file_path = dir_path + "fold_" + str(i) + "/" + str(topic_id) + ".txt"
            (train_x, train_y) = generate_threads(file_path,
                                                  mv,
                                                  train_x,
                                                  train_y)

    print "generate valid dataset"
    # generate valid dataset
    for topic_id in xrange(n_topics):
        file_path = dir_path + "fold_" + str(valid_indicator) + "/" + str(topic_id) + ".txt"
        (valid_x, valid_y) = generate_threads(file_path,
                                              mv,
                                              valid_x,
                                              valid_y)

    print "generate test dataset"
    # generate test dataset
    for topic_id in xrange(n_topics):
        file_path = dir_path + "fold_" + str(test_indicator) + "/" + str(topic_id) + ".txt"
        (test_x, test_y) = generate_threads(file_path,
                                            mv,
                                            test_x,
                                            test_y)
    global ERROR_FIND
    global SUCCESS_FIND
    print "word_vector not found: %d" %(ERROR_FIND)
    print "word_vector found: %d"%(SUCCESS_FIND)
    return (train_x, train_y, valid_x, valid_y, test_x, test_y)


"""
================ END microblog data process ============================================
"""





def gen_structured_xy(thread_data):
    """
    """
    train_x = []
    train_y = []
    x_term = []
    y_term = []
    for sen_dict in thread_data:
        for k, v in sen_dict.items():
            x_term.append(v[2])
            y_term.append(v[0])
        train_x.append(x_term)
        train_y.append(y_term)
        x_term = []
        y_term = []
    return train_x, train_y

def train_sen_flatten(thread_data):
    """
    """
    sens = []
    for sen_dict in thread_data:
        for k, v in sen_dict.items():
            sens.append(v[1])
    return sens
