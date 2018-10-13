# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'bi_daf'
    model_save = model_name + '_2'
    is_bn = True
    epoch = 5
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 100
    encoder_layer_num = 1
    dropout_p = 0.3
    val_every = 100
    val_mean = False  # 这个指标用来衡量，是否是每隔固定次数验证一次

    # 联合训练
    is_for_rouge = True
    if is_for_rouge:
        criterion = 'RougeLoss'
        lamda = 0.01
        val_mean = True

    # 测试
    model_test = 'bi_daf_3_mrt'
    is_true_test = True

config = Config()
