# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'qa_net'
    model_save = model_name + '_1'  # merge 500
    epoch = 12
    batch_size = 12
    hidden_size = 128  # 必须为8的倍数
    dropout_p = 0.1
    encoder_dropout_p = 0.1

    val_every = 100

    # 联合训练
    is_for_rouge = False
    if is_for_rouge:
        criterion = 'RougeLoss'
        lamda = 5

    # 测试
    model_test = 'qa_net_1'
    is_true_test = False

config = Config()
