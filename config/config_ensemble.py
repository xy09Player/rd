# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    model_test = 'ensemble'

    match_lstm_plus_1 = 'match_lstm_plus_1'  # 0.9047
    match_lstm_plus_1_mrt = 'match_lstm_plus_1_mrt'  # 0.9032
    match_lstm_plus_2 = 'match_lstm_plus_2'  # 0.9048
    match_lstm_plus_2_mrt = 'match_lstm_plus_2_mrt'  # 0.9032
    match_lstm_plus_3 = 'match_lstm_plus_3'  # 0.9039
    match_lstm_plus_3_mrt = 'match_lstm_plus_3_mrt'  # 0.9016

    r_net_1 = 'r_net_1'  # 0.9046
    r_net_1_mrt = 'r_net_1_mrt'  # 0.9035
    r_net_2 = 'r_net_2'  # 0.9049
    r_net_2_mrt = 'r_net_2_mrt'  # 0.9036
    r_net_3 = 'r_net_3'  # 0.9020

    bi_daf_1 = 'bi_daf_1'  # 0.9004
    bi_daf_1_mrt = 'bi_daf_1_mrt'  # 0.9031
    bi_daf_2 = 'bi_daf_2'  # 0.902
    bi_daf_2_mrt = 'bi_daf_2_mrt'  # 0.9006
    bi_daf_3 = 'bi_daf_3'  # 0.9028
    bi_daf_3_mrt = 'bi_daf_3_mrt'  # 0.9025

    m_reader_plus_1 = 'm_reader_plus_1'  # 0.9054
    m_reader_plus_1_mrt = 'm_reader_plus_1_mrt'  # 0.9045
    m_reader_plus_2 = 'm_reader_plus_2'  # 0.9039
    m_reader_plus_2_mrt = 'm_reader_plus_2_mrt'  # 0.904
    m_reader_plus_3 = 'm_reader_plus_3'  # 0.9048
    m_reader_plus_3_mrt = 'm_reader_plus_3_mrt'  # 0.9045

    m_reader_1 = 'm_reader_1'  # 0.9015
    m_reader_1_mrt = 'm_reader_1_mrt'  # 0.9026
    m_reader_2 = 'm_reader_2'  # 0.9014
    m_reader_2_mrt = 'm_reader_2_mrt'  # 0.9011
    m_reader_3 = 'm_reader_3'  # 0.901
    m_reader_3_mrt = 'm_reader_3_mrt'  # 0.9005

    # model_lst = [
    #     match_lstm_plus_1,
    #     match_lstm_plus_2,
    #     match_lstm_plus_3,
    #     r_net_1,
    #     r_net_2,
    #     r_net_3,
    #     bi_daf_1_mrt,
    #     bi_daf_2,
    #     bi_daf_3,
    #     m_reader_plus_1,
    #     m_reader_plus_2_mrt,
    #     m_reader_plus_3,
    #     m_reader_1_mrt,
    #     m_reader_2,
    #     m_reader_3
    # ]
    # model_weight = [0.9047, 0.9048, 0.9039,
    #                 0.9046, 0.9049, 0.902,
    #                 0.9031, 0.902, 0.9028,
    #                 0.9054, 0.904, 0.9048,
    #                 0.9026, 0.9014, 0.901]

    model_lst = [
        m_reader_plus_1,
        m_reader_1_mrt
    ]
    model_weight = [0.9054,
                    0.9026]

    model_weight = utils.mean(model_weight)

    is_true_test = True

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
