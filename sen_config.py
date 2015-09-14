# -*- coding: utf-8 -*-
"""
rnn_params = {
    "fold": 5,
    "lr": 0.0627,
    "win": 5, # number of words in context window if use context-window
    "em_dimension":100 # the dimension of word embeding
}
"""
MIN_SENTENCE_LENGTH = 3

RNN_PARAMS = {

    "dh": 10,
    "nc": 3,
    "ne": 14326,
    "de": 100,
    "sw": 7,
    "lr": 0.0627
}

POLARITY = {
    "[-3]": -3,
    "[-2]": -2,
    "[-1]": -1,
    "[+1]": 1,
    "[+2]": 2,
    "[+3]": 3,
    "[1]": 1,
    "[2]": 2,
    "[3]": 3,
    "neg": -1,
    "neu": 0,
    "mix": 0,
    "nr": 0,
    "pos": 1
    }

FILE_PATH = {
    "dict": "../dict/word_dict.txt",
    "reviewdata1": "../data/Reviews-9-products/Apex_AD2600_Progressive_scan_DVD_player.txt",
    "reviewdata2": "../data/Reviews-9-products/Cannon_G3.txt",
    "reviewdata3": "../data/Reviews-9-products/Canon_PowerShot_SD500.txt",
    "reviewdata4": "../data/Reviews-9-products/Canon_S100.txt",
    "reviewdata5": "../data/Reviews-9-products/Creative_Labs_Nomad_Jukebox_Zen_Xtra_40GB.txt",
    "reviewdata6": "../data/Reviews-9-products/Diaper_Champ.txt",
    "reviewdata7": "../data/Reviews-9-products/Hitachi_router.txt",
    "reviewdata8": "../data/Reviews-9-products/ipod.txt",
    "reviewdata9": "../data/Reviews-9-products/Linksys_Router.txt",
    "reviewdata10": "../data/Reviews-9-products/MicroMP3.txt",
    "reviewdata11": "../data/Reviews-9-products/Nikon_coolpix_4300.txt",
    "reviewdata12": "../data/Reviews-9-products/Nokia_6600.txt",
    "reviewdata13": "../data/Reviews-9-products/Nokia_6610.txt",
    "reviewdata14": "../data/Reviews-9-products/norton.txt",
    "mddata": "../data/mddata.txt"
    }
