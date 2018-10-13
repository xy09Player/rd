# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'match_lstm'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 10
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 150
    encoder_layer_num = 1
    dropout_p = 0.2
    val_every = 100

    embedding_type = 'expand'  # standard

    # 联合训练
    is_for_rouge = False
    if is_for_rouge:
        criterion = 'RougeLoss'
    lamda = 2  # 5

    # 测试
    model_test = 'match_lstm_1'
    is_true_test = True

config = Config()
