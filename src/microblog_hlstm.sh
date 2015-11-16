#!/bin/bash
cmd="python main.py
    --experiment=microblog
    --model_name=slstm_tlstm_model
    --level1_model_name=slstm_model
    --level2_model_name=tlstm_model
    --dataset_name=microblog
    --log_path=../logs/weibo_hlstm_gpu_1.txt
    --word_dim=200
    --level1_input=200
    --level1_hidden=200
    --level2_input=200
    --level2_hidden=200
    --n_output=3
    --cnn_n_feature_maps=100
    --cnn_window_sizes=2@3@4
    --dropout_rate=0.5
    --optimizer_method=adadelta
    --learning_rate=1
    --batch_type=minibatch
    --batch_size=200
    --n_epochs=100
    --train_pos=0@1@2
    --valid_pos=3
    --test_pos=4
    --valid_frequency=1000"

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