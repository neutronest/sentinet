# -*- coding: utf-8 -*-
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import corpora, models
import pdb

class Vectorize(object):
    """
    basic word to vector class

    Used for transfer the original words and sentences to the word vector
    and sentence vector


    """

    def __init__(self):
        """
        Parameters(self):
        -----------------
        model: the model of word2vec
               type: Word2Vec

        size: the dimensionality of word feature vectors
              default value is 300,
              type: int

        window: max distance between the current
                and predicted word within a sentence
                default value is 15
                type: int

        workers: number of cpu cores
                 default value: 4
                 type: int

        dictionary: TODO
        """
        self.word2vec_model = None
        self.doc2vec_model = None
        self.google_model = None
        self.size = 300
        self.window = 15
        self.min_count = 5
        self.workers = 4
        self.alpha = 0.025
        self.dictionary = None
        return

    def train_google_model(self, google_file):
        """
        using the google word vector dataset to extract the word feature

        Parameters:
        -----------
        google_file: the location about google.bin/G.bin
                     type: string

        Return:
        -------
        None

        """
        self.google_modopel = Word2Vec.load_word2vec_format(google_file, binary=True)
        return

    def get_google_vector(self, word):
        """
        """
        return self.google_model[word]

    def word_vector_train(self, sentences):
        """
        train the model with the sentences corpus

        Parameters:
        -----------
        sentences: the corpus
                   type:
        """
        self.word2vec_model = Word2Vec(sentences, self.size,
                              self.window,
                              self.min_count,
                              self.workers)
        return

    def doc_vector_train(self, sentences):
        """
        """
        sen_sig_pre = "sen_"
        sen_cur = 0
        doc_sens = []

        len_of_sen = len(sentences)
        for i in xrange(len_of_sen):
            sen_sig = sen_sig_pre + str(sen_cur)
            labeled_sentence = LabeledSentence(words=sentences[i], labels=[sen_sig])
            doc_sens.append(labeled_sentence)
            sen_cur += 1


        self.doc2vec_model = Doc2Vec(alpha=0.025,
                                     min_alpha=0.025,
                                     size=self.size)
        self.doc2vec_model.build_vocab(doc_sens)

        # train
        for epoch in xrange(10):
            self.doc2vec_model.train(doc_sens)
            self.alpha -= 0.002
            self.doc2vec_model.min_alpha = self.alpha
        return



    def get_sentence_vector(self, sentence):
        """
        """
        assert self.doc2vec_model != None
        return


    def word_vector_save(self, filename):
        """
        """
        self.model.save(filename)
        return

    def word_vector_load(self, filename):
        """
        """
        self.model = Word2Vec.load(filename)
        return

    def word_vector_get(self, word):
        """
        """
        return self.word2vec_model[word]

    def doc_vector_get(self, sen_sig):
        """
        """
        return self.doc2vec_model[sen_sig]

    def dict_init(self, sentences):
        """
        """
        assert(sentences != None)
        self.dictionary = corpora.Dictionary(sentences)
        print "the corpus's dictionary init success."
        return

    def get_bow_vector(self, sentence):
        """
        get corresponding index tuple from a sentence.

        Parameters:
        -----------
        sentence: a word list (just a sentence)
                  type: str list

        Reutrn:
        -------
        bow_vector: a tuple that denote word's index and frequncy
                    type: tuple
        """
        assert(self.dictionary != None)
        bow_vector = self.dictionary.doc2bow(sentence)
        return bow_vector

    def get_word_index(self, sentence):
        """
        """
        bow_vector = self.get_bow_vector(sentence)
        word_indexs = []
        for t in bow_vector:
            word_indexs.append(t[0])
        return word_indexs

    def get_token2id(self):
        return self.dictionary.token2id

    def print_token2id(self):
        for (k, v) in self.dictionary.token2id.items():
            print k, v

        return
