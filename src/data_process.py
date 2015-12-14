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



def cosine_sim(A_ids, B_ids):
    """
    cosine similarity calculate
    """
    sum_ab = 0.0
    a_sqrt = 0.0
    b_sqrt = 0.0

    a_sqrt = math.sqrt(sum([i*i for i in A_ids]))
    b_sqrt = math.sqrt(sum([i*i for i in B_ids]))
    sum_ab = sum([ ai * bi for ai, bi in zip(A_ids, B_ids)])

    return sum_ab / (a_sqrt * 1. * b_sqrt)

def generate_feature(json_dict,
                     feature_name,
                     feature,
                     parent,
                     grandpa,
                     edge_type="equal"):
    """

    """
    d_feature = [0, 0]
    if edge_type == "equal":
        d_feature[0] = 1 if parent != -1 and \
                       feature == json_dict[parent][feature_name] \
                       else 0
        d_feature[1] = 1 if grandpa != -1 and \
                       feature == json_dict[grandpa][feature_name] \
                       else 0
    elif edge_type == "contain":
        d_feature[0] = 1 if parent != -1 and \
                       len(set(feature).intersection(set(json_dict[parent][feature_name]))) > 0 \
                       else 0
        d_feature[1] = 1 if grandpa != -1 and \
                       len(set(feature).intersection(set(json_dict[grandpa][feature_name]))) > 0 \
                       else 0
    elif edge_type == "similarity":
        cur_word_id = feature
        pa_word_id = json_dict[parent][feature_name]
        grad_word_id = json_dict[grandpa][feature_name]

        cur_word_fix_id = [1 if i in cur_word_id else 0 for i in xrange(33024)]
        pa_word_fix_id = [1 if i in pa_word_id else 0 for i in xrange(33024)]
        grad_word_fix_id = [1 if i in grad_word_id else 0 for i in xrange(33024)]
        d_feature[0] = 1 if parent != -1 and \
                       cosine_sim(cur_word_fix_id, pa_word_fix_id) > 0.1 \
                       else 0
        d_feature[1] = 1 if grandpa != -1 and \
                       cosine_sim(cur_word_fix_id, grad_word_fix_id) > 0.1 \
                       else 0

    else:
        print "foobar"
    return d_feature


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
    features = ["polarity",
                "author",
                "similarity",
                "emoji",
                "hashtag",
                "mention"]

    content_dict = OrderedDict()

    with open(file_path, "r") as train_ob:
        for line in train_ob:
            line = line.strip()
            line_json = json.loads(line)
            # parse the component
            threadid = str(line_json['threadid'])
            docid = int(line_json['docid'])
            parent = int(line_json['parent'])
            words = line_json['words']
            # IMPORTANT! change -1, 0, 1 to 0, 1, 2
            label = int(line_json['label'])+1
            emoji = line_json['emoji']
            author = line_json['username']
            mention = line_json['mention']
            hashtag = line_json['hashtag']
            text = line_json['text']

            # prepare words_ids
            words_ids = [words_table[word] for word in words \
                         if words_table.get(word) != None]

            if len(words_ids) < 6:
                # none word in the weibo
                # copy it's parent's content
                #words_ids = copy.deepcopy(data_x[threadid][0][parent])
                words_ids += [0] * (6 - len(words_ids))
                NONE_WORD_NUM += 1


            if content_dict.get(threadid) != None:

                content_dict[threadid].append({'docid': docid,
                                               'parent': parent,
                                               'label': label,
                                               'emoji': emoji,
                                               'author': author,
                                               'mention': mention,
                                               'hashtag': hashtag,
                                               'wordsids': words_ids})
            else:
                content_dict[threadid] = [{
                    'docid': docid,
                    'parent': parent,
                    'label': label,
                    'emoji': emoji,
                    'author': author,
                    'mention': mention,
                    'hashtag': hashtag,
                    'wordsids': words_ids
                }]
            # prepare features for RNN
            # polarity
            d_polarity = [0, 0]
            # check if parent and grandpa exist

            grandpa = content_dict[threadid][parent]['parent'] if parent != -1 else -1

            # features
            #d_polarity = generate_feature(content_dict[threadid], "label", label, parent, grandpa, "equal")
            d_author = generate_feature(content_dict[threadid], "author", author, parent, grandpa, "equal")
            d_emoji = generate_feature(content_dict[threadid], "emoji", emoji, parent, grandpa, "contain")
            d_hashtag = generate_feature(content_dict[threadid], "hashtag", hashtag, parent, grandpa, "contain")
            d_mention = generate_feature(content_dict[threadid], "mention", mention, parent, grandpa, "contain")
            # similar text
            #d_similar = generate_feature(content_dict[threadid], "wordsids", words_ids, parent, grandpa, "similarity")
            d_t = d_author + d_emoji + d_hashtag + d_mention# + d_similar
            # applying data
            if data_x.get(threadid) == None:
                # new thread
                max_len = len(words_ids)
                #relation_seq = [docid, parent]
                # TODO: change relation encoding!
                # in relation seq, add empty node as guard node
                relation_seq = [0]
                mask = [1] * len(words_ids)
                data_x[threadid] = [[words_ids], max_len, [mask],  relation_seq, [d_t], [text]]
            else:
                data_x[threadid][0].append(words_ids)
                max_len = len(words_ids) if max_len < len(words_ids) else max_len
                mask = [1] * len(words_ids)
                data_x[threadid][1] = max_len
                data_x[threadid][2].append(mask)
                data_x[threadid][3].append(parent)
                data_x[threadid][4].append(d_t)
                data_x[threadid][5].append(text)
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


def get_pos_neg_words():
    """
    """
    pos_words = []
    neg_words = []
    with open("../dict/pos_words.txt", "r") as pos_ob, open("../dict/neg_words.txt", "r") as neg_ob:
        for pos_line, neg_line in zip(pos_ob, neg_ob):
            pos_word = pos_line.strip().decode("utf-8")
            neg_word = neg_line.strip().decode("utf-8")
            pos_words.append(pos_word)
            neg_words.append(neg_word)
    pos_ob.close()
    neg_ob.close()

    return pos_words, neg_words


def generate_diffdata(test_x, test_y):
    """
    revdata is the data that polarity token and label is different
    """
    pos_words, neg_words = get_pos_neg_words()
    revdata = []
    idx = 0
    acc_idx = 0
    for (tx, item_x), (tx, item_y) in zip(test_x.items(), test_y.items()):

        texts = item_x[5]
        ys = item_y
        for i in xrange(len(texts)):
            pos_cnt = sum([1 for w in pos_words if w in texts[i]])
            neg_cnt = sum([1 for w in neg_words if w in texts[i]])

            if (pos_cnt > neg_cnt and ys[i] == 0) or \
               (pos_cnt < neg_cnt and ys[i] == 2):
                revdata.append({"idx": acc_idx + i,
                                "text": texts[i],
                                "label": ys[i]})
                print idx
                idx += 1
        acc_idx += len(texts)
    return revdata

if __name__ == "__main__":

    root_same_polarity_res = 0.
    root_same_polarity_acc = 0.
    n_weibo = 0
    pa_same_polarity_res = 0.
    pa_same_polarity_acc = 0.


    words_table, lookup_table, wordid_acc = build_lookuptable()
    (train_x, train_y, valid_x, valid_y, test_x, test_y) \
        = load_microblogdata([0,1,2], 3, 4, words_table)
    revdata = generate_diffdata(test_x, test_y)
    pdb.set_trace()
    """
    for (train_threadid_x, train_item_x), (train_threadid_y, train_item_y) in \
        zip(train_x.items(), train_y.items()):
        assert(train_threadid_x == train_threadid_y)
        # prepare train data
        ids_matrix = np.asarray(train_item_x[0],
                                dtype=np.int32)
        # test
        input_x = np.transpose(np.asarray(ids_matrix,
                                             dtype=np.int32))
        mask = np.transpose(np.asarray(train_item_x[2],
                                       dtype=theano.config.floatX))
        input_y = np.asarray([ [1 if i == y else 0 for i in xrange(3)]  for y in train_item_y],
                             dtype=np.int32)

        label_y = np.asarray(train_item_y,
                             dtype=np.int32)
        relations = np.asarray(train_item_x[3],
                               dtype=np.int32)

        h0 = np.asarray(np.zeros((len(relations), 100),
                                 dtype=theano.config.floatX))
        c0 = np.asarray(np.zeros((len(relations), 100),
                                 dtype=theano.config.floatX))
        th_init = np.asarray(np.zeros(100*(len(relations)+1),
                                      dtype=theano.config.floatX))
        tc_init = np.asarray(np.zeros(100*(len(relations)+1),
                                      dtype=theano.config.floatX))
        dt = np.asarray(train_item_x[4],
                        dtype=theano.config.floatX)

        yt = np.asarray([[0, 0, 0]] + \
                        [[1 if i == y else 0 for i in xrange(3)]  for y in train_item_y],
                        dtype=theano.config.floatX)
        yt_pred = np.asarray(np.zeros_like(yt),
                             dtype=theano.config.floatX)


    for (train_threadid_x, train_item_x), (train_threadid_y, train_item_y) in \
        zip(train_x.items(), train_y.items()):
        # statistics
        label_y = np.asarray(train_item_y,
                             dtype=np.int32)
        root_pol = label_y[0]
        root_same_polarity_acc += sum([1 for y in label_y[1:] if y == root_pol])
        relations = train_item_x[3]
        for i in xrange(len(relations)):
            print relations
            print i
            print label_y
            if i == 0:
                continue
            if label_y[i-1] == label_y[relations[i]]:
                pa_same_polarity_acc += 1
        n_weibo += len(label_y)

    for (valid_threadid_x, valid_item_x), (valid_threadid_y, valid_item_y) in \
        zip(valid_x.items(), valid_y.items()):
        label_y = np.asarray(valid_item_y, dtype=np.int32)
        root_pol = label_y[0]
        root_same_polarity_acc += sum([1 for y in label_y[1:] if y == root_pol])
        #print label_y, sum([1 for y in label_y[1:] if y == root_pol])
        relations = valid_item_x[3]
        for i in xrange(len(relations)):
            if i == 0:
                continue
            if label_y[i-1] == label_y[relations[i]]:
                pa_same_polarity_acc += 1
        n_weibo += len(label_y)

    for (test_threadid_x, test_item_x), (test_threadid_y, test_item_y) in \
        zip(test_x.items(), test_y.items()):

        label_y = np.asarray(test_item_y,
                             dtype=np.int32)
        root_pol = label_y[0]
        root_same_polarity_acc += sum([1 for y in label_y[1:] if y == root_pol])
        #print label_y, sum([1 for y in label_y[1:] if y == root_pol])
        relations = test_item_x[3]
        for i in xrange(len(relations)):
            if i == 0:
                continue
            if label_y[i-1] == label_y[relations[i]]:
                pa_same_polarity_acc += 1
        n_weibo += len(label_y)

    print n_weibo
    print "[=== same polarity with root weibo statistics ===]"
    print root_same_polarity_acc * 1. / n_weibo
    print pa_same_polarity_acc * 1. / n_weibo
    """
