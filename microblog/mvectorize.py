# -*- coding=utf-8 -*-
import vec_config
import json
from gensim import corpora, models
from gensim.models.word2vec import Word2Vec
import word_cutting
import mutils
import pdb
import sys

sys.path.append("../src")
import config
"""
Return Function:
----------------

character_to_vectorion

get_dictionary

"""
class MVectorize(object):

    def __init__(self):

        word_cutting.load_thirdparty_words("../dict/favourate.txt")
        word_cutting.load_thirdparty_words("../dict/emoji.txt")
        self.dictionary = None
        self.word_pesg_tbl = {} # the word-pesg hashtable
        self.words_doc = None
        self.tfidf = None
        self.words_model = None
        return

    def gen_words(self, filepath):
        """
        generate basic words from original data file
        """
        lines = mutils.get_line_from_file(filepath)
        texts = mutils.get_text_only_from_lines(lines)
        text_filters = []
        emoji_filters = []
        for text in texts:
            text_filter = ""
            emoji_list, text_filter = word_cutting.filter_emoji_from_textV2(text)
            mention_list, text_filter = word_cutting.filter_syntax_from_textV2(text_filter, '@')
            hashtag_list, text_filter = word_cutting.filter_syntax_from_textV2(text_filter, '#')
            text_filters.append(text_filter)
            if len(emoji_list) != 0:
                emoji_filters.append(emoji_list)

        words_doc = []

        for text in text_filters:
            words = word_cutting.cut_directly(text)
            words_doc.append(words)
        """ TRICKS!!! """
        return words_doc

    def addtional_corpus(self, file_path):
        """
        """
        return


    def gen_words_vector(self, file_path):
        """
        """
        words_doc = self.gen_words(file_path)
        words_num = len([w for words in words_doc for w in words])
        print "[---the words_num of model is %d---]"%(words_num)
        print "[---generate word vectors model!---]"
        self.words_model = Word2Vec(words_doc, size=config.options['word_dim'], window=10, min_count=1, workers=4)
        print "[--- word embedding model Done! ---]"
        return

    def gen_words_doc(self, file_path):
	"""
	"""
        words_doc = self.gen_words(file_path)
        # filter stop words
        stop_words = word_cutting.get_stopwords()
        ## test
        #words_doc = [[word for word in doc if word.encode("utf-8") not in stop_words] for doc in words_doc]
        self.words_doc = words_doc
        return

    def dict_init_from_file(self):
        """
        """
        # words_doc = [[] for doc in words_doc]
        self.dict_init_from_texts(self.words_doc)
        return

    def dict_init_from_texts(self, words_texts):
        """ change all words in the texts to numeric vector

        Parameters:
        -----------
        words_texts: two dimension list
               type: str list list

        Return:
        -------
        dictionary: the dictionary of corpora
                    type: gensim.corpora.dictionary.Dictionary
        """
        self.dictionary = corpora.Dictionary(words_texts)
        self.dictionary.save(vec_config.dictionary_path)
        # print self.dictionary
        return self.dictionary

    def get_token2id(self):
        return self.dictionary.token2id

    def print_token2id(self):
        for (k, v) in self.dictionary.token2id.items():
            #print type(k), type(v)
            print k.encode("utf-8"), v
        return

    def get_bow_vector(self, words):
        """

        """
        assert(self.dictionary != None)
        bow_vector = self.dictionary.doc2bow(words)
        return bow_vector


if __name__ == "__main__":
    dir_path = "../data/weibo/fold_data/"
    mv = MVectorize()
    mv.gen_words_vector("../data/weibo/weiboV2.tsv")
    pdb.set_trace()
