# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    ensemble_name = 'ensemble_1'

    match_lstm_plus_1 = 'match_lstm_plus_1'  # 0.911
    match_lstm_plus_1_mrt = 'match_lstm_plus_1_mrt'  # 0.9111
    match_lstm_plus_2 = 'match_lstm_plus_2'  # 0.9132
    match_lstm_plus_2_mrt = 'match_lstm_plus_2_mrt'  # 0.9123
    match_lstm_plus_3 = 'match_lstm_plus_3'  # 0.9082
    match_lstm_plus_3_mrt = 'match_lstm_plus_3_mrt'  # 0.912

    r_net_1 = 'r_net_1'  # 0.9129
    r_net_1_mrt = 'r_net_1_mrt'  # 0.9135
    r_net_2 = 'r_net_2'  # 0.9097
    r_net_2_mrt = 'r_net_2_mrt'  # 0.9111
    r_net_3 = 'r_net_3'  # 0.9114
    r_net_3_mrt = 'r_net_3_mrt'  # 0.9107

    bi_daf_1 = 'bi_daf_1'  # 0.9094
    bi_daf_1_mrt = 'bi_daf_1_mrt'  # 0.9126
    bi_daf_2 = 'bi_daf_2'  # 0.9108
    bi_daf_2_mrt = 'bi_daf_2_mrt'  # 0.9107
    bi_daf_3 = 'bi_daf_3'  # 0.9065
    bi_daf_3_mrt = 'bi_daf_3_mrt'  # 0.9097

    m_reader_plus_1 = 'm_reader_plus_1'  # 0.9098
    m_reader_plus_1_mrt = 'm_reader_plus_1_mrt'  # 0.9128
    m_reader_plus_2 = 'm_reader_plus_2'  # 0.9113
    m_reader_plus_2_mrt = 'm_reader_plus_2_mrt'  # 0.9132
    m_reader_plus_3 = 'm_reader_plus_3'  # 0.9111
    m_reader_plus_3_mrt = 'm_reader_plus_3_mrt'  # 0.9117

    m_reader_1 = 'm_reader_1'  # 0.9093
    m_reader_1_mrt = 'm_reader_1_mrt'  # 0.9087
    m_reader_2 = 'm_reader_2'  # 0.911
    m_reader_2_mrt = 'm_reader_2_mrt'  # 0.9106
    m_reader_3 = 'm_reader_3'  # 0.9092
    m_reader_3_mrt = 'm_reader_3_mrt'  # 0.9094

    model_lst = [
        match_lstm_plus_1_mrt,
        match_lstm_plus_2,
        match_lstm_plus_3_mrt,
        r_net_1_mrt,
        r_net_2_mrt,
        r_net_3,
        bi_daf_1_mrt,
        bi_daf_2,
        bi_daf_3_mrt,
        m_reader_plus_1_mrt,
        m_reader_plus_2_mrt,
        m_reader_plus_3_mrt,
        m_reader_1,
        m_reader_2,
        m_reader_3_mrt
    ]
    model_weight = [0.9111, 0.9132, 0.912,
                    0.9135, 0.9111, 0.9114,
                    0.9126, 0.9108, 0.9097,
                    0.9128, 0.9132, 0.9117,
                    0.9093, 0.911, 0.9094]

    # model_lst = [
    #     match_lstm_plus_2,
    #     r_net_1_mrt,
    #     bi_daf_1_mrt,
    #     m_reader_plus_2_mrt,
    #     m_reader_2,
    # ]
    # model_weight = [0.9111, 0.9132, 0.912,
    #                 0.9135, 0.9111]

    # model_weight = utils.softmax(model_weight)
    model_weight = utils.mean(model_weight)

    is_true_test = True

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
