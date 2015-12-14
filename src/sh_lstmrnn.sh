#!/bin/bash

for layersize in 64 128
do
    for dropout in false dropout
    do
        for batchsize in 10 20
        do
            cmd="python main.py
    --experiment=microblog
    --model_name=slstm_trnn_model
    --level1_model_name=slstm_model
    --level2_model_name=trnn_model
    --dataset_name=microblog
    --log_path=../logs/LSTMRNN/$layersize-$dropout-$batchsize.txt
    --word_dim=128
    --level1_input=128
    --level1_hidden=$layersize
    --level2_input=$layersize
    --level2_hidden=$layersize
    --n_output=3
    --cnn_n_feature_maps=100
    --cnn_window_sizes=2@3@4
    --if_dropout=$dropout
    --optimizer_method=adadelta
    --learning_rate=1
    --batch_type=minibatch
    --batch_size=$batchsize
    --n_epochs=100
    --train_pos=0@1@2
    --valid_pos=3
    --test_pos=4
    --valid_frequency=200"
            THEANO_FLAGS="mode=FAST_RUN,floatX=float32" $cmd &
        done
    done
done