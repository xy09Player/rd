# coding = utf-8
# author = xy

import os
import time
import json
import pickle
from data_pre import title_question
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from my_metrics import blue
from my_metrics import rouge_test
import loader
import utils
import preprocess_data
from config import config_base
from config import config_r_net
from config import config_match_lstm
from config import config_match_lstm_plus
from config import config_bi_daf
from config import config_qa_net
from config import config_m_reader
from config import config_m_reader_plus
from config import config_ensemble
from modules import match_lstm
from modules import match_lstm_plus
from modules import r_net
from modules import bi_daf
from modules import qa_net
from modules import m_reader
from modules import m_reader_plus

# config = config_match_lstm.config
# config = config_match_lstm_plus.config
# config = config_r_net.config
# config = config_bi_daf.config
# config = config_m_reader.config
config = config_m_reader_plus.config
# config = config_ensemble.config


def test(gen_result):
    time0 = time.time()

    # prepare
    if config.is_true_test:
        preprocess_data.gen_pre_file_for_test()

    # load w2v
    embedding_np_train = loader.load_w2v(config.train_embedding + '.npy')
    if config.is_true_test:
        embedding_np_test = loader.load_w2v(config.test_embedding + '.npy')

    # prepare: test_df
    if config.is_true_test and (os.path.isfile(config.test_df) is False):
        preprocess_data.gen_test_datafile(config.test_data, config.test_df)

    if (config.is_true_test is False) and (os.path.isfile(config.test_val_df) is False):
        preprocess_data.gen_test_datafile(config.val_data, config.test_val_df)

    # load data
    if config.is_true_test is False:
        if os.path.isfile(config.test_val_pkl):
            with open(config.test_val_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = loader.load_data(config.test_val_df, config.train_vocab_path, config.tag_path)
            with open(config.test_val_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    else:
        if os.path.isfile(config.test_pkl):
            with open(config.test_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = loader.load_data(config.test_df, config.test_vocab_path, config.tag_path)
            with open(config.test_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    # build test dataloader
    test_loader = loader.build_loader(
        dataset=test_data[:6],
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False
    )

    # model initial
    param = {
        'embedding': embedding_np_train,
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': config.encoder_dropout_p,
        'encoder_bidirectional': config.encoder_bidirectional,
        'encoder_layer_num': config.encoder_layer_num,
        'is_bn': config.is_bn
    }

    model = eval(config.model_name).Model(param)

    # load model param, and training state
    model_path = os.path.join('model', config.model_test)
    print('load model, ', model_path)
    state = torch.load(model_path)
    model.load_state_dict(state['best_model_state'])

    # 改变embedding_fix
    if config.is_true_test:
        model.embedding.sd_embedding.embedding_fix = nn.Embedding(
            num_embeddings=embedding_np_test.shape[0],
            embedding_dim=embedding_np_test.shape[1],
            padding_idx=0,
            _weight=torch.Tensor(embedding_np_test)
        )
        model.embedding.sd_embedding.embedding_fix.weight.requires_grad = False
        model.embedding.sd_embedding.vocab_size = embedding_np_test.shape[0]
    model = model.cuda()

    best_loss = state['best_loss']
    best_epoch = state['best_epoch']
    best_step = state['best_step']
    best_time = state['best_time']
    use_time = state['time']
    print('best_epoch:%2d, best_step:%5d, best_loss:%.4f, best_time:%d, use_time:%d' %
          (best_epoch, best_step, best_loss, best_time, use_time))

    # gen result
    result_start = []
    result_end = []
    result_start_p = []
    result_end_p = []

    model.eval()
    with torch.no_grad():
        cc = 0
        cc_total = len(test_loader)
        print('total iter_num:%d' % cc_total)
        for batch in test_loader:
            # cuda, cut
            batch = utils.deal_batch(batch)
            outputs = model(batch)
            start, end = utils.answer_search(outputs)

            start = start.reshape(-1).cpu().numpy().tolist()
            end = end.reshape(-1).cpu().numpy().tolist()

            result_start = result_start + start
            result_end = result_end + end

            start_p = outputs[0].cpu().numpy().tolist()
            end_p = outputs[1].cpu().numpy().tolist()

            result_start_p += start_p
            result_end_p += end_p

            cc += 1
            if cc % 100 == 0:
                print('processing: %d/%d' % (cc, cc_total))

    # 需要生成结果
    if gen_result:
        if config.is_true_test:
            df = pd.read_csv(config.test_df)
        else:
            df = pd.read_csv(config.test_val_df)

        # 生成str结果
        titles = df['title']
        shorten_content = df['shorten_content']
        question = df['question']
        assert len(titles) == len(shorten_content) == len(result_start) == len(result_end)
        result = utils.gen_str(titles, shorten_content, question, result_start, result_end, add_liangci=config.is_true_test)

        # gen a submission
        if config.is_true_test:
            articled_ids = df['article_id'].astype(str).values.tolist()
            question_ids = df['question_id'].values
            submission = []
            temp_a_id = articled_ids[0]
            temp_qa = []
            for a_id, q_id, a in zip(articled_ids, question_ids, result):
                if a_id == temp_a_id:
                    sub = {'questions_id': q_id, 'answer': a}
                    temp_qa.append(sub)
                else:
                    submission.append({'article_id': temp_a_id, 'questions': temp_qa})
                    temp_a_id = a_id
                    temp_qa = [{'questions_id': q_id, 'answer': a}]
            submission.append({'article_id': temp_a_id, 'questions': temp_qa})

            submission_article = [s['article_id'] for s in submission]
            submission_questions = [s['questions'] for s in submission]
            submission_dict = dict(zip(submission_article, submission_questions))

            with open(config.test_data, 'r') as file:
                all_data = json.load(file)
            all_article = [d['article_id'] for d in all_data]

            submission = []
            for a_id in all_article:
                if a_id in submission_dict:
                    submission.append({'article_id': a_id, 'questions': submission_dict[a_id]})
                else:
                    submission.append({'article_id': a_id, 'questions': []})

            with open(config.submission, mode='w', encoding='utf-8') as f:
                json.dump(submission, f, ensure_ascii=False)

        # my_metrics
        if config.is_true_test is False:
            answer_true = df['article_answer'].values
            assert len(result) == len(answer_true)
            blue_score = blue.Bleu()
            rouge_score = rouge_test.RougeL()
            for a, r in zip(answer_true, result):
                if a == a:
                    blue_score.add_inst(r, a)
                    rouge_score.add_inst(r, a)
            print('rouge_L score: %.4f, blue score:%.4f' % (rouge_score.get_score(), blue_score.get_score()))

        # to .csv
        if config.is_true_test is False:
            df['answer_pred'] = result
            df['answer_start_pred'] = result_start
            df['answer_end_pred'] = result_end
            csv_path = os.path.join('result', config.model_test+'_val.csv')
            df.to_csv(csv_path, index=False)

    # save result_ans_range
    if config.is_true_test:
        save_path = os.path.join('result/ans_range', config.model_test+'_submission.pkl')
    else:
        save_path = os.path.join('result/ans_range', config.model_test+'_val.pkl')

    result_ans_range = {'start_p': result_start_p, 'end_p': result_end_p}
    torch.save(result_ans_range, save_path)
    print('time:%d' % (time.time()-time0))


def test_ensemble():
    time0 = time.time()

    if config.is_true_test:
        df = pd.read_csv(config.test_df)
    else:
        df = pd.read_csv(config.test_val_df)

    # 加权求和
    model_lst = config.model_lst
    model_weight = config.model_weight
    start_p = np.zeros([len(df), config.max_len])
    end_p = np.zeros([len(df), config.max_len])
    print('model number:%d' % (len(model_lst)))
    for ml, mw in zip(model_lst, model_weight):
        result_path = os.path.join('result/ans_range', ml)
        ans_range = torch.load(result_path)
        s_p = ans_range['start_p']
        e_p = ans_range['end_p']
        start_p += np.array(s_p) * mw
        end_p += np.array(e_p) * mw

    start_p = torch.from_numpy(start_p)
    end_p = torch.from_numpy(end_p)
    dataset = data.TensorDataset(start_p, end_p)
    p_loader = data.DataLoader(
        dataset=dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False
    )

    result_start = []
    result_end = []
    for s_p, e_p in p_loader:
        s_e_p = torch.stack([s_p, e_p])
        s, e = utils.answer_search(s_e_p)
        result_start += s.reshape(-1).numpy().tolist()
        result_end += e.reshape(-1).numpy().tolist()

    # 生成str结果
    titles = df['title']
    shorten_content = df['shorten_content']
    question = df['question']
    assert len(titles) == len(shorten_content) == len(result_start) == len(result_end)
    result = utils.gen_str(titles, shorten_content, question, result_start, result_end, add_liangci=config.is_true_test)

    # gen a submission
    if config.is_true_test:
        articled_ids = df['article_id'].astype(str).values.tolist()
        question_ids = df['question_id'].values
        submission = []
        temp_a_id = articled_ids[0]
        temp_qa = []
        for a_id, q_id, a in zip(articled_ids, question_ids, result):
            if a_id == temp_a_id:
                sub = {'questions_id': q_id, 'answer': a}
                temp_qa.append(sub)
            else:
                submission.append({'article_id': temp_a_id, 'questions': temp_qa})
                temp_a_id = a_id
                temp_qa = [{'questions_id': q_id, 'answer': a}]
        submission.append({'article_id': temp_a_id, 'questions': temp_qa})

        submission_article = [s['article_id'] for s in submission]
        submission_questions = [s['questions'] for s in submission]
        submission_dict = dict(zip(submission_article, submission_questions))

        with open(config.test_data, 'r') as file:
            all_data = json.load(file)
        all_article = [d['article_id'] for d in all_data]

        submission = []
        for a_id in all_article:
            if a_id in submission_dict:
                submission.append({'article_id': a_id, 'questions': submission_dict[a_id]})
            else:
                submission.append({'article_id': a_id, 'questions': []})

        with open(config.submission, mode='w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False)

    # my_metrics
    if config.is_true_test is False:
        answer_true = df['answer'].values
        assert len(result) == len(answer_true)
        blue_score = blue.Bleu()
        rouge_score = rouge_test.RougeL()
        for a, r in zip(answer_true, result):
            blue_score.add_inst(r, a)
            rouge_score.add_inst(r, a)
        print('rouge_L score: %.4f, blue score:%.4f' % (rouge_score.get_score(), blue_score.get_score()))

    # to .csv
    if config.is_true_test is False:
        df['answer_pred'] = result
        df['answer_start_pred'] = result_start
        df['answer_end_pred'] = result_end

        df = df[['article_id', 'title', 'content', 'question', 'answer', 'answer_pred',
                 'answer_start', 'answer_end', 'answer_start_pred', 'answer_end_pred']]
        csv_path = os.path.join('result', config.ensemble_name + '_val.csv')
        df.to_csv(csv_path, index=False)

    print('time:%d' % (time.time()-time0))


def test_ensemble_fix():
    time0 = time.time()

    if config.is_true_test:
        df = pd.read_csv(config.test_df)
    else:
        df = pd.read_csv(config.test_val_df)

    # 确定每个模型的 s, e, p值
    model_num = len(config.model_lst)
    ensemble_result = [[] for _ in range(model_num)]
    for model, e_result in zip(config.model_lst, ensemble_result):
        result_path = os.path.join('result/ans_range', model)
        ans_range = torch.load(result_path)
        s_p = ans_range['start_p']
        e_p = ans_range['end_p']
        s_p = torch.from_numpy(np.array(s_p))
        e_p = torch.from_numpy(np.array(e_p))

        dataset = data.TensorDataset(s_p, e_p)
        p_loader = data.DataLoader(
            dataset=dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            drop_last=False
        )

        for ssss_p, eeee_p in p_loader:
            s_e_p = torch.stack([ssss_p, eeee_p])
            s, e, p = utils.answer_search(s_e_p, return_p=True)
            for ss, ee, pp in zip(s, e, p):
                e_result.append([ss, ee, pp])

    # 确定所有模型的最优解
    result_start = []
    result_end = []
    for i in range(len(ensemble_result[0])):
        ss = 0
        ee = 0
        vv = 0
        for j in range(model_num):
            ss_tmp = ensemble_result[j][i][0]
            ee_tmp = ensemble_result[j][i][1]
            vv_tmp = ensemble_result[j][i][2]

            if vv_tmp > vv:
                vv = vv_tmp
                ss = ss_tmp
                ee = ee_tmp

        result_start.append(ss)
        result_end.append(ee)

    # 生成str结果
    titles = df['title']
    shorten_content = df['shorten_content']
    question = df['question']
    assert len(titles) == len(shorten_content) == len(result_start) == len(result_end)
    result = utils.gen_str(titles, shorten_content, question, result_start, result_end, add_liangci=config.is_true_test)

    # gen a submission
    if config.is_true_test:
        articled_ids = df['article_id'].astype(str).values.tolist()
        question_ids = df['question_id'].values
        submission = []
        temp_a_id = articled_ids[0]
        temp_qa = []
        for a_id, q_id, a in zip(articled_ids, question_ids, result):
            if a_id == temp_a_id:
                sub = {'questions_id': q_id, 'answer': a}
                temp_qa.append(sub)
            else:
                submission.append({'article_id': temp_a_id, 'questions': temp_qa})
                temp_a_id = a_id
                temp_qa = [{'questions_id': q_id, 'answer': a}]
        submission.append({'article_id': temp_a_id, 'questions': temp_qa})

        submission_article = [s['article_id'] for s in submission]
        submission_questions = [s['questions'] for s in submission]
        submission_dict = dict(zip(submission_article, submission_questions))

        with open(config.test_data, 'r') as file:
            all_data = json.load(file)
        all_article = [d['article_id'] for d in all_data]

        submission = []
        for a_id in all_article:
            if a_id in submission_dict:
                submission.append({'article_id': a_id, 'questions': submission_dict[a_id]})
            else:
                submission.append({'article_id': a_id, 'questions': []})

        with open(config.submission, mode='w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False)

    # my_metrics
    if config.is_true_test is False:
        answer_true = df['answer'].values
        assert len(result) == len(answer_true)
        blue_score = blue.Bleu()
        rouge_score = rouge_test.RougeL()
        for a, r in zip(answer_true, result):
            blue_score.add_inst(r, a)
            rouge_score.add_inst(r, a)
        print('rouge_L score: %.4f, blue score:%.4f' % (rouge_score.get_score(), blue_score.get_score()))

    # to .csv
    if config.is_true_test is False:
        df['answer_pred'] = result
        df['answer_start_pred'] = result_start
        df['answer_end_pred'] = result_end

        df = df[['article_id', 'title', 'content', 'question', 'answer', 'answer_pred',
                 'answer_start', 'answer_end', 'answer_start_pred', 'answer_end_pred']]
        csv_path = os.path.join('result', config.ensemble_name + '_val.csv')
        df.to_csv(csv_path, index=False)

    print('time:%d' % (time.time()-time0))


if __name__ == '__main__':
    if config == config_ensemble.config:
        print('ensemble...')
        test_ensemble()
        # test_ensemble_fix()
    else:
        print('single model...')
        gen_result = config.gen_result
        test(gen_result)
