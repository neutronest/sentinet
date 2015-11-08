python main.py \
    --experiment=microblog \
    --model_name=srnn_trnn_model \
    --dataset_name=microblog \
    --log_path=../logs/microblog_srnn_trnn.txt  \
    --word_dim=300 \
    --level1_input=300 \
    --level1_hidden=300 \
    --level1_output=3 \
    --level2_input=300 \
    --level2_hidden=300 \
    --level2_output=3 \
    --dropout_rate=0.5 \
    --optimizer_method=sgd \
    --learning_rate=0.01 \
    --batch_type=minibatch \
    --batch_size=200 \
    --n_epochs=50 \
    --train_pos=0@1@2@3 \
    --valid_pos=4 \
    --test_pos=4 \
    --valid_frequency=1000
