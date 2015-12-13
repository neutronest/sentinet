#!/bin/bash
cmd_1="python baseline.py
    --experiment=microblog
    --model_name=cnn_model
    --level1_model_name=cnn_model
    --level2_model_name=trnn_model
    --dataset_name=microblog
    --log_path=../logs/CNN/cnn_128_100_345_t.txt
    --word_dim=128
    --level1_input=128
    --level1_hidden=100
    --level2_input=100
    --level2_hidden=100
    --n_output=3
    --cnn_n_feature_maps=100
    --cnn_window_sizes=3@4@5
    --if_dropout=true
    --optimizer_method=adadelta
    --learning_rate=1
    --batch_size=10
    --n_epochs=100
    --train_pos=0@1@2
    --valid_pos=3
    --test_pos=4
    --valid_frequency=200"

cmd_2="python baseline.py
    --experiment=microblog
    --model_name=cnn_model
    --level1_model_name=cnn_model
    --level2_model_name=trnn_model
    --dataset_name=microblog
    --log_path=../logs/CNN/cnn_128_100_345_f.txt
    --word_dim=128
    --level1_input=128
    --level1_hidden=100
    --level2_input=100
    --level2_hidden=100
    --n_output=3
    --cnn_n_feature_maps=100
    --cnn_window_sizes=3@4@5
    --if_dropout=false
    --optimizer_method=adadelta
    --learning_rate=1
    --batch_size=10
    --n_epochs=100
    --train_pos=0@1@2
    --valid_pos=3
    --test_pos=4
    --valid_frequency=40"

THEANO_FLAGS="mode=FAST_RUN,floatX=float32" $cmd_1 &

THEANO_FLAGS="mode=FAST_RUN,floatX=float32" $cmd_2 &
