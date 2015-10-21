# -*- coding: utf-8 -*-
import pdb
from collections import OrderedDict
from vectorize import Vectorize
import logging
import numpy as np
import theano
"""

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
    load_utterance_dataset(1000, 1005)
