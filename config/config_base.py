# coding = utf-8
# author = xy


class ConfigBase:
    train_data = 'data/first_question.json'
    test_data = 'data/公布测试集-0919.json'

    max_len = 500
    train_df = 'data_gen/train_df.csv'
    val_df = 'data_gen/val_df.csv'
    test_val_df = 'data_gen/test_val_df.csv'
    test_df = 'data_gen/test_df.csv'

    train_pkl = 'data_gen/train_df.pkl'
    val_pkl = 'data_gen/val_df.pkl'
    test_val_pkl = 'data_gen/test_val_df.pkl'
    test_pkl = 'data_gen/test_df.pkl'

    train_vocab_path = 'data_gen/train_vocab.pkl'
    test_vocab_path = 'data_gen/test_vocab.pkl'
    tag_path = 'data_gen/tag2index.pkl'

    pre_embedding_zh = 'data/merge_sgns_bigram_char300.txt'
    pre_embedding_en = 'data/glove300.txt'
    train_embedding = 'data_gen/train_embedding'
    test_embedding = 'data_gen/test_embedding'

    submission = 'submission/result.txt'

    criterion = 'MyNLLLoss'
    is_bn = True
    lr = 1e-4
    epoch = 10
    mode = 'GRU'
    batch_size = 32
    test_batch_size = 64
    hidden_size = 75
    encoder_layer_num = 2
    encoder_bidirectional = True
    encoder_dropout_p = 0.1
    dropout_p = 0.2
    max_grad = 5
    val_every = 100

config = ConfigBase()
