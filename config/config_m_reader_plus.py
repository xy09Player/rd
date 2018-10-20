# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'm_reader_plus'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 15
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 100
    encoder_layer_num = 1
    dropout_p = 0.2
    val_every = 100
    val_mean = False

    val_split_value = 1.1

    # 联合训练
    is_for_rouge = False
    if is_for_rouge:
        criterion = 'RougeLoss'
        lamda = 0.01
        val_mean = True

    # 测试
    model_test = 'm_reader_plus_1'
    gen_result = True
    is_true_test = False
    test_batch_size = 64

config = Config()
