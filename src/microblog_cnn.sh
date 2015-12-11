#!/bin/bash
cmd="python baseline.py
    --experiment=microblog
    --model_name=cnn_model
    --dataset_name=microblog
    --log_path=../logs/v1_cnn.txt
    --word_dim=100
    --n_output=3
    --cnn_n_feature_maps=100
    --cnn_window_sizes=2@3@4
    --if_dropout=false
    --optimizer_method=adadelta
    --learning_rate=1
    --batch_size=10
    --n_epochs=100
    --train_pos=0@1@2
    --valid_pos=3
    --test_pos=4
    --valid_frequency=200"

if [ $# -eq 0 ]; then
    THEANO_FLAGS="mode=FAST_RUN,floatX=float32" $cmd

elif [ "$1" = "gpu" ]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
    THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" $cmd
else
    echo "error"
fi