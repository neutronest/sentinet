python main.py --experiment=swda --model_name=rcnn --dataset_name=swda \
 --log_path=../logs/swda_test.txt --word_dim=300 --cnn_n_feature_maps=100 \
 --cnn_window_sizes=2@3@4 --rnn_n_hidden=300 --rnn_n_out=43 --dropout_rate=0.5 \
 --optimizer_method=sgd --learning_rate=0.01 --batch_size=10 \
 --n_epochs=100 --train_pos=1000 --valid_pos=10101