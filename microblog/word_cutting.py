#-*- coding:utf-8 -*-
import json
import jieba
import jieba.posseg as pseg
import data_filter
import re
from parse import *
import vec_config
import jieba.posseg as pseg
import pdb


def load_thirdparty_words(filepath):
    """
    """
    jieba.load_userdict(filepath)
    return


def cut_directly(text):
    """ cut words to list from jieba,
        do not filt the stopwords
        do not delete the repeat words

    Parameters:
    -----------
    texts:

    """
    text = text.decode("utf-8")
    seg_list = jieba.cut(text, cut_all=False)
    return [w for w in seg_list]

def cut(text):
    """ cut words to list from jieba

    Parameters:
    -----------
    texts: the single sentence.
           type: str

    Returns:
    --------
    list(seg_list):
        type: str list
    """
    seg_list = cut_directly(text)
    # YOU CAN CHOOSE IF USE N_GRAM HERE
    #bi_seg_list = [text[i:i+2] for i in xrange(len(text)-1)]
    #tri_seg_list = [text[i:i+3] for i in xrange(len(text)-2)]
    #seg_list = list(seg_list) + bi_seg_list + tri_seg_list
    #seg_list = list(set(seg_list))
    stop_words = get_stopwords()
    #seg_list = [w for w in seg_list if w.encode("utf-8") not in stop_words]
    return seg_list

def bigram(seg_list):
    """
    """
    seg_bigram = []
    len_of_seg = len(seg_list)
    if len_of_seg < 2:
        return seg_list
    for i in xrange(len_of_seg-1):
        bi_word = ''.join(seg_list[i:i+2])
        seg_bigram.append(bi_word)
    return seg_bigram

def cut_with_pseg(text):

    seg_list = pseg.cut(text)
    stop_words = get_stopwords()
    seg_list = [w for w in seg_list if w.word.encode("utf-8") not in stop_words]

    return list(seg_list)

""" ================== emoji mention and hashtag process
"""



def get_stopwords():
    """ get the stopwords list from file

    Parameters:
    -----------
    None

    Return:
    -------
    stopwords: the stopwords string
               type: str

    """
    stopwords_list = []
    with open(vec_config.stopwords_path) as file_ob:
        for line in file_ob:
            stopwords_list.append(line)
        file_ob.close()

    return ' '.join(stopwords_list)


def get_emoji():
    """
    Parameters:
    -----------
    None

    Returns:
    --------
    emoji_list:
                type: str list



    """
    emoji_list = []
    with open(vec_config.emoji_path, "r") as file_ob:
        for emoji in file_ob:
            # delete \n
            emoji = emoji.strip("\n")
            emoji_list.append(emoji)
    file_ob.close()

    return emoji_list

def filter_emoji_from_text(text):
    """
    """
    emoji_list = get_emoji()
    emoji_res = []
    # get all emoji that appeared
    # not delete the emoji in this turn, for store the repeate emoji
    for emoji in emoji_list:

        if emoji in text:
            emoji_res.append(emoji)

    # delete all emoji from origin text
    text_filter = text
    for emoji in emoji_res:
        text_filter = text_filter.replace(emoji, "")


    return emoji_res, text_filter

def filter_emoji_from_textV2(text):
    """
    """
    emoji_res = ["["+str(r.fixed[0])+"]" for r in findall("[{}]", text)]
    text_filter = text
    for emoji in emoji_res:
        # DO NOT DELETE the EMOJI DIRECTLY
        text_filter = text_filter.replace(emoji, " "+emoji+" ")
    return emoji_res, text_filter

def filter_syntax_from_textV2(text, syntax='@'):
    """
    """
    syntax_res = []
    text_filter = text
    if syntax == '@':
        syntax_res = [r.fixed[0] for r in findall("@{}:", text)]
        syntax_res = syntax_res + [r.fixed[0] for r in findall("@{} ", text)]
        syntax_res = syntax_res + [r.fixed[0] for r in findall("@{}\n", text)]
        syntax_res = syntax_res + [r.fixed[0] for r in findall("@{}(", text)]
        syntax_res = syntax_res + [r.fixed[0] for r in findall("@{})", text)]
        syntax_res = syntax_res + [r.fixed[0] for r in findall("@{}（", text)]
        syntax_res = syntax_res + [r.fixed[0] for r in findall("@{}）]", text)]
    elif syntax == '#':
        syntax_res = [r.fixed[0] for r in findall("#{}#", text)]
    else:
        syntax_res = []
    for s in syntax_res:
        text_filter = text.replace(s, "")
    text_filter = text_filter.replace(syntax, "")
    return syntax_res, text_filter



def filter_syntax_from_text(text, syntax='@'):
    """
    Parameters:
    -----------
    text: one line string
          type: str

    syntax: the mark parse character
            type: char
    """
    mention_list = []
    stop_syntax = [':', '(', ')', ' ', '（', '）']
    c_flag = False
    mention = ""
    for c in text:
        # find the mattching begin position
        if c == syntax and c_flag == False:
            c_flag = True
            continue
        # durning the mention getter
        if c_flag == True:
            if syntax == '#':
                if c != syntax:
                    mention = mention + str(c)
                else:
                    # end mention
                    # delete the syntax char
                    mention_list.append(mention.decode("utf-8"))
                    # delete mention from origin text
                    text = text.replace(mention, "")
                    # re_init
                    mention = ""
                    c_flag = False
            elif syntax == '@':
                # if c != ' ' and c != ':' and c != syntax:
                if c not in stop_syntax and c != syntax:
                    mention = mention + str(c)
                else:
                    # end mention
                    # delete the syntax char
                    mention_list.append(mention.decode("utf-8"))
                    # delete mention from origin text
                    text = text.replace(mention, "")
                    # re_init
                    mention = ""
                    if c in stop_syntax:
                        c_flag = False
                    else: # c = @
                        c_flag = True

    """
    if mention_list != []:
        print syntax
        for mention in mention_list:
            print mention
    """
    return mention_list, text

def get_weibos():
    regex = re.compile(r"\b(\w+)\s*:\s*([^:]*)(?=\s+\w+\s*:|$)")
    weibo_list = []
    with open("../data/mahang_dat.txt") as file_ob:
        next(file_ob)
        for line in file_ob:
            items = dict(regex.findall(line))
            word = items.get("text")
            if word:
                weibo_list.append(word.decode("utf-8"))
    return weibo_list


if __name__ == "__main__":

    input_str = raw_input()
    while input_str != "EOF":
        seg_list = cut_directly(input_str)
        print " ".join(seg_list)
        # next input
        input_str = raw_input()
