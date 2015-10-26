python main.py --experiment=microblog --model_name=rcnn_onestep_model --dataset_name=microblog \
    --log_path=../logs/microblog_test.txt --word_dim=300 --cnn_n_feature_maps=100 \
    --cnn_window_sizes=2@3@4 --rnn_n_hidden=300 --rnn_n_out=3 --dropout_rate=0.5 \
    --optimizer_method=sgd --learning_rate=0.01 --batch_type=one --batch_size=1 \
    --n_epochs=10 --train_pos=0@1@2 --valid_pos=3 --test_pos=4 \
    --valid_frequency=1000
