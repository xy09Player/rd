# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    model_test = 'ensemble'

    match_lstm_plus_1 = 'match_lstm_plus_1'  # 0.9047
    match_lstm_plus_2 = 'match_lstm_plus_2'  # 0.9048

    r_net_1 = 'r_net_1'  # 0.9046
    r_net_2 = 'r_net_2'  # 0.9049

    bi_daf_1_mrt = 'bi_daf_1_mrt'  # 0.9031
    bi_daf_3 = 'bi_daf_3'  # 0.9028

    m_reader_plus_1 = 'm_reader_plus_1'  # 0.9054
    m_reader_plus_3 = 'm_reader_plus_3'  # 0.9048

    m_reader_1_mrt = 'm_reader_1_mrt'  # 0.9026
    m_reader_2 = 'm_reader_2'  # 0.9014

    model_lst = [
        match_lstm_plus_1,
        match_lstm_plus_2,
        r_net_1,
        r_net_2,
        bi_daf_1_mrt,
        bi_daf_3,
        m_reader_plus_1,
        m_reader_plus_3,
        m_reader_1_mrt,
        m_reader_2

    ]
    model_weight = [0.9047, 0.9048,
                    0.9046, 0.9049,
                    0.9031, 0.9028,
                    0.9054, 0.9048,
                    0.9026, 0.9014]
    model_weight = utils.mean(model_weight)

    is_true_test = False

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
