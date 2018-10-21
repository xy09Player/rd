# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    model_test = 'ensemble'

    bi_daf_1 = 'bi_daf_1'  # 0.9004
    bi_daf_1_mrt = 'bi_daf_1_mrt'  # 0.9031

    m_reader_plus_1 = 'm_reader_plus_1'  # 0.9054
    m_reader_plus_1_mrt = 'm_reader_plus_1_mrt'  # 0.9045

    m_reader_1 = 'm_reader_1'  # 0.9015
    m_reader_1_mrt = 'm_reader_1_mrt'  # 0.9026

    model_lst = [
        bi_daf_1_mrt,
        m_reader_plus_1,
        m_reader_1_mrt
    ]
    model_weight = [0.9031, 0.9054, 0.9026]
    model_weight = utils.mean(model_weight)

    is_true_test = False

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
