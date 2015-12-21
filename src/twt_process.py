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
import math
sys.path.append("../microblog")
from mvectorize import MVectorize
from collections import OrderedDict


def build_lookup_table():
    """
    build the tweet lookup table
    """
    mv = MVectorize()
    mv.gen_google_vector(config.options['google_news_data'])
    #pdb.set_trace()
    words_table = {}
    lookup_table = []
    lookup_table.append([0] * config.options["google_dim"])
    wordid_acc = 1
    with open(config.options['twt_train_data'], "r") as twt_train_ob, \
         open(config.options['twt_test_data'], "r") as twt_test_ob:
        for line in twt_train_ob:
            if line == "\n":
                continue
            line_arr = line.strip().split("\t")
            words = line_arr[2].split(" ")
            words = filter(lambda x: x in mv.google_model.vocab, words)
            for word in words:
                if words_table.get(word) is None:
                    words_table[word] = wordid_acc
                    lookup_table.append(mv.google_model[word])
                    wordid_acc += 1
        for line in twt_test_ob:
            if line == "\n":
                continue
            line_arr = line.strip().split("\t")
            words = line_arr[2].split(" ")
            words = filter(lambda x: x in mv.google_model.vocab, words)
            for word in words:
                if words_table.get(word) is None:
                    words_table[word] = wordid_acc
                    lookup_table.append(mv.google_model[word])
                    wordid_acc += 1
    twt_train_ob.close()
    twt_test_ob.close()
    return words_table, lookup_table, wordid_acc

def generate_twt(words_table, file_path):
    """
    loading the all twt data
    """
    twts = []
    twt_thread = {"username":[],
                  "mask": [],
                  "wordid": [],
                  "label": [],
                  "max_len": 0}
    with open(file_path, "r") as file_ob:
        for line in file_ob:
            # new
            if line == "\n" and len(twt_thread) != 0:
                twts.append(twt_thread)
                twt_thread =  {"username":[],
                               "mask": [],
                               "wordid": [],
                               "label": [],
                               "max_len": 0}

            line_arr = line.strip().split("\t")
            thread_id = line_arr[0]
            username = line_arr[1]
            words = line_arr[2].split(" ")
            label = int(line_arr[3])

            words_ids = [words_table[word] for word in words \
                         if words_table.get(word) != None]
            if len(words_ids) < config.options['min_word_one_sen']:
                words_ids.append([0] * (config.options['min_word_one_sen'-len(words_ids)]))

            mask = [1] * len(words_ids)
            twt_thread['username'].append(username)
            twt_thread['wordid'].append(words_ids)
            twt_thread['mask'].append(mask)
            twt_thread['label'].append(label)
            if twt_thread['max_len'] < len(words_ids):
                twt_thread['max_len'] = len(words_ids)
    twts.append(twt_thread)
    # padding
    for thread in twts:
        max_len = thread['max_len']
        ids = thread['wordid']
        masks = thread['mask']
        for wordid, mask in zip(ids, masks):
            if len(mask) < max_len:
                padding_len = max_len - len(mask)
                wordid += [0] * padding_len
                mask += [0] * padding_len
    return twts

def load_twt_data(words_table):
    """

    """
    train_twts = generate_twt(words_table, config.options['twt_train_data'])
    test_twts = generate_twt(words_table, config.options['twt_test_data'])
    return (train_twts, test_twts)

if __name__ == "__main__":
    words_table, lookup_table, wordid_acc = build_lookup_table()
    (train_twts, test_twts) = load_twt_data(words_table)
    pdb.set_trace()
    exit()
