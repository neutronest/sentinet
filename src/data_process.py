# -*- coding: utf-8 -*-
import pdb
from collections import OrderedDict
from vectorize import Vectorize
import logging
import numpy as np
import theano
import json
import sys
import config
import copy
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



NONE_WORD_ID = 0
ERROR_FIND = 0
SUCCESS_FIND = 0
SHORT_SENS_FIND = 0
NONE_WORD_NUM = 0
LESS_WORD_NUM = 0

def build_lookuptable():

    dir_path = "../data/weibo/fold_data/"
    mv = MVectorize()
    mv.gen_words_vector("../data/weibo/weiboV2.tsv")
    # build lookup table
    words_table = {}
    lookup_table = []
    # add none word representation
    lookup_table.append([0] * config.options['word_dim'])
    wordid_acc = 1
    for i in [0, 1, 2, 3, 4]:
        for topic_id in xrange(51):
            file_path = dir_path + "fold_" + str(i) + "/" + str(topic_id) + ".txt"
            words_table, lookup_table, wordid_acc = update_lookuptable(file_path,
                                                                      mv,
                                                                      words_table,
                                                                      lookup_table,
                                                                      wordid_acc)

    return words_table, lookup_table, wordid_acc

def update_lookuptable(file_path,
                      mvectorize,
                      words_table,
                      lookup_table,
                      wordid_acc):
    """
    """
    with open(file_path, "r") as file_ob:
        for line in file_ob:
            line = line.strip()
            line_json = json.loads(line)
            words = line_json['words']
            #print "fiter the unseen words"
            words = filter(lambda x: x in mvectorize.words_model.vocab, words)
            for word in words:
                if words_table.get(word) is None:
                    words_table[word] = wordid_acc
                    lookup_table.append(mvectorize.words_model[word])
                    wordid_acc += 1

    return words_table, lookup_table, wordid_acc


def generate_threadsV2(file_path,
                       words_table,
                       n_hidden,
                       data_x,
                       data_y):
    """ get the well-sturctured data of train/valid/test Version 2...

    Parameters:
    -----------
    file_path: the specific file path
        type: str

    mvectorize: the mvectorize object
        type: mvectorize

    n_hidden: the num of hidden state in the proposed model

    data_x: the X of train/valid/test
        type: {threadid:[word_embs_flatten,
                         [[sen_start_pos, sen_end_pos],..],
                         {h_state_docid:h_state_vector},
                         [[docid, parentid],...]}
              threadid: str
              word_embs_flatten: concatenate all word embs of each sentence.
                                 the length is equal to the len(word_emb) * sum(len(each sen))
              type: ndarray of list

              sen_start_pos: int, the start position of each sen in word_embs_flatten
              sen_end_pos: int, the end position of each sen in word_embs_flatten
              type: (int, int) of list

              docid: the current id of sen
              parentid: the corresponding parent id
              type: (int, int) of list

    data_y: the Y of train/valid/test
          type: {threadid: [(docid, label)]}

    """
    global NONE_WORD_NUM
    global LESS_WORD_NUM
    with open(file_path, "r") as train_ob:
        for line in train_ob:
            line = line.strip()
            line_json = json.loads(line)
            # parse the conponent
            threadid = str(line_json['threadid'])
            docid = int(line_json['docid'])
            parent = int(line_json['parent'])
            words = line_json['words']
            # IMPORTANT! change -1, 0, 1 to 0, 1, 2
            label = int(line_json['label'])+1

            words_ids = [words_table[word] for word in words \
                         if words_table.get(word) != None]

            if len(words_ids) == 0:
                # none word in the weibo
                # copy it's parent's content
                #words_ids = copy.deepcopy(data_x[threadid][0][parent])
                words_ids = [0]
                NONE_WORD_NUM += 1

            # applying data
            if data_x.get(threadid) == None:
                # new thread
                max_len = len(words_ids)
                relation_seq = [docid, parent]
                mask = [1] * len(words_ids)
                data_x[threadid] = [[words_ids], max_len, [mask],  [relation_seq]]
            else:
                data_x[threadid][0].append(words_ids)
                max_len = len(words_ids) if max_len < len(words_ids) else max_len
                mask = [1] * len(words_ids)
                data_x[threadid][1] = max_len
                data_x[threadid][2].append(mask)
                data_x[threadid][3].append([docid, parent])
            if data_y.get(threadid) == None:
                data_y[threadid] = [label]
            else:
                data_y[threadid].append(label)
    #PADDING PROCESS
    for k, v in data_x.items():
        # v[0] is the 2d matrix
        # v[1] is max_len
        # v[2] is relation_pairs
        max_len = v[1]
        for ids, mask in zip(v[0], v[2]):
            if len(ids) < max_len:
                padding_len = max_len - len(ids)
                ids += [NONE_WORD_ID] * padding_len
                mask += [0] * padding_len
    return (data_x, data_y)

def load_microblogdata(train_indicators,
                       valid_indicator,
                       test_indicator,
                       words_table=None):
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

    # build lookup table
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
            (train_x, train_y) = generate_threadsV2(file_path,
                                                    words_table,
                                                    config.options['word_dim'],
                                                    train_x,
                                                    train_y)

    print "generate valid dataset"
    # generate valid dataset
    for topic_id in xrange(n_topics):
        file_path = dir_path + "fold_" + str(valid_indicator) + "/" + str(topic_id) + ".txt"
        (valid_x, valid_y) = generate_threadsV2(file_path,
                                                words_table,
                                                config.options['word_dim'],
                                                valid_x,
                                                valid_y)

    print "generate test dataset"
    # generate test dataset
    for topic_id in xrange(n_topics):
        file_path = dir_path + "fold_" + str(test_indicator) + "/" + str(topic_id) + ".txt"
        (test_x, test_y) = generate_threadsV2(file_path,
                                              words_table,
                                              config.options['word_dim'],
                                              test_x,
                                              test_y)
    global ERROR_FIND
    global SUCCESS_FIND
    global SHORT_SENS_FIND
    global NONE_WORD_NUM
    print "word_vector not found: %d" %(ERROR_FIND)
    #print "word_vector found: %d"%(SUCCESS_FIND)
    print "short sens find: %d"%(SHORT_SENS_FIND)
    print "less word num: %d"%(LESS_WORD_NUM)
    print "none word num: %d"%(NONE_WORD_NUM)
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


if __name__ == "__main__":
    (train_x, train_y, valid_x, valid_y, test_x, test_y) \
        = load_microblogdata([0,1,2], 3, 4)

    for data_x in test_x.items():
        ids_matrix = data_x[0]
        for ids in ids_matrix:
            for word_id in ids:
                if word_id == 33025:
                    print ids
