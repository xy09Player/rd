# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'r_net'
    model_save = model_name + '_3'
    is_bn = True
    epoch = 15
    mode = 'GRU'
    batch_size = 32
    hidden_size = 75
    encoder_layer_num = 1
    dropout_p = 0.3
    val_every = 100
    val_mean = False

    # 联合训练
    is_for_rouge = False
    if is_for_rouge:
        criterion = 'RougeLoss'
        lamda = 0.01
        val_mean = True

    # 测试
    model_test = 'r_net_3_mrt'
    is_true_test = True

config = Config()
