# encoding = utf-8
# author = xy

import torch
import jieba
from jieba import posseg
import nltk
import numpy as np
import pickle
from data_pre import title_question
import pandas as pd


def pad(data_array, length):
    """ padding """
    tmp = []
    for d in data_array:
        if len(d) > length:
            tmp.append(d[: length])
        elif len(d) < length:
            tmp.append(d + [0] * (length - len(d)))
        else:
            tmp.append(d)
    data_array = tmp
    return data_array


def deal_batch_original(batch):
    """
    deal batch: cuda, cut
    :param batch:[content, question, start, end] or [content, question]
    :return: batch_done
    """

    def cut(indexs):
        max_len = get_mask(indexs).sum(dim=1).max().item()
        max_len = int(max_len)
        return indexs[:, :max_len]

    is_training = True if len(batch) == 4 else False
    if is_training:
        contents, questions, starts, ends = batch
    else:
        contents, questions = batch

    # cuda
    contents = contents.cuda()
    questions = questions.cuda()
    if is_training:
        starts = starts.cuda()
        ends = ends.cuda()

    # cut
    contents = cut(contents)
    questions = cut(questions)

    if is_training:
        return [contents, questions, starts, ends]
    else:
        return [contents, questions]


def deal_batch(batch):
    """
    deal batch: cuda, cut
    :param batch:[content_index, content_flag, content_is_in_title, content_is_in_question, question_index,
    question_flag, start, end] or [content_index, content_flag, content_is_in_title, content_is_in_question,
    question_index, question_flag]
    :return: batch_done
    """

    def cut(data):
        max_len = get_mask(data[0]).sum(dim=1).max().item()
        max_len = int(max_len)
        data = [d[:, :max_len] for d in data]
        return data

    def padding(data, length):
        cur_len = data[0].size(1)
        if cur_len > length:
            data = [d[:, :length] for d in data]
        elif cur_len < length:
            pad_len = length - cur_len
            batch_size = data[0].size(0)
            pad_zeros = data[0].new_zeros(batch_size, pad_len)
            data = [torch.cat([d, pad_zeros], dim=1) for d in data]

        return data

    contents = batch[: 3]
    questions = batch[3: 6]
    is_training = True if len(batch) == 8 else False

    # cuda
    contents = [c.cuda() for c in contents]
    questions = [q.cuda() for q in questions]
    if is_training:
        starts = batch[6].cuda()
        ends = batch[7].cuda()

    # cut
    # contents = cut(contents)
    # questions = cut(questions)

    # padding
    # contents = padding(contents, 500)
    # questions = padding(questions, 150)

    if is_training:
        return [*contents, *questions, starts, ends]
    else:
        return [*contents, *questions]


def get_mask(tensor, padding_idx=0):
    """ get mask tensor """
    return torch.ne(tensor, padding_idx).float()


def masked_flip(seq_tensor, mask):
    """
     flip seq_tensor
    :param seq_tensor: (seq_len, batch_size, input_size)
    :param mask: (batch_size, seq_len)
    :return: (seq_len, batch_size, input_size)
    """
    length = mask.eq(1).long().sum(dim=1)
    batch_size = seq_tensor.size(1)

    outputs = []
    for i in range(batch_size):
        temp = seq_tensor[:, i, :]
        temp_length = length[i]

        idx = list(range(temp_length - 1, -1, -1)) + list(range(temp_length, seq_tensor.size(0)))
        idx = seq_tensor.new_tensor(idx, dtype=torch.long)

        temp = temp.index_select(0, idx)
        outputs.append(temp)

    outputs = torch.stack(outputs, dim=1)
    return outputs


def answer_search(answer_prop, return_p=False):
    """
     global search best answer for model predict
    :param answer_prop: (2, batch_size, c_len)
    :return: ans_s, ans_e
    """
    batch_size = answer_prop.size(1)
    c_len = answer_prop.size(2)

    ans_s_p = answer_prop[0]
    ans_e_p = answer_prop[1]
    b_zero = answer_prop.new_zeros(batch_size, 1)

    ans_s_e_p_lst = []
    for i in range(c_len):
        temp_ans_s_e_p = ans_s_p * ans_e_p
        ans_s_e_p_lst.append(temp_ans_s_e_p)

        ans_s_p = ans_s_p[:, :(c_len - 1)]
        ans_s_p = torch.cat([b_zero, ans_s_p], dim=1)

    ans_s_e_p = torch.stack(ans_s_e_p_lst, dim=2)

    # get the best end position, and move steps
    max_prop1, max_prop_idx1 = torch.max(ans_s_e_p, 1)
    max_prop2, max_prop_idx2 = torch.max(max_prop1, 1)

    ans_e = max_prop_idx1.gather(1, max_prop_idx2.unsqueeze(1)).squeeze(1)
    ans_s = ans_e - max_prop_idx2

    if return_p:
        ans_s = ans_s.numpy().reshape(-1).tolist()
        ans_e = ans_e.numpy().reshape(-1).tolist()
        answer_prop = answer_prop.numpy()
        answer_start_prop = answer_prop[0]
        answer_end_prop = answer_prop[1]
        value = []
        for a_s, a_e, s_p, e_p in zip(ans_s, ans_e, answer_start_prop, answer_end_prop):
            value.append(s_p[a_s] * e_p[a_e])
        return ans_s, ans_e, value
    else:
        return ans_s, ans_e


def softmax(weight):
    exp = np.exp(weight)
    return exp / exp.sum()


def mean(weight):
    weight = np.array(weight)
    weight = weight / sum(weight)
    return weight


def _rouge_score(start_y, end_y, start_pred, end_pred, gamma):
    """ 计算给定区间的(1-rouge)  """
    start = max(start_y, start_pred)
    end = min(end_y, end_pred)

    interval = end - start + 1
    if interval <= 0:
        return 1
    else:
        length_pred = end_pred - start_pred + 1
        length_y = end_y - start_y + 1
        prec = interval / length_pred if length_pred > 0 else 0
        rec = interval / length_y if length_y > 0 else 0

        if prec != 0 and rec != 0:
            score = 1 - ((1 + gamma ** 2) * prec * rec) / (rec + gamma ** 2 * prec)
        else:
            score = 1
        return score


def rouge_scores(start_y, end_y, start_pro, end_pro, gamma):
    """ 计算某一条记录的期望rouge """
    result = 0
    for s in range(end_y + 1):
        for j in range(start_y, len(start_pro)):
            result += _rouge_score(start_y, end_y, s, j, gamma) * start_pro[s] * end_pro[j]

    return result


def deal_data(titles, contents, questions):
    """
     index, tag, is_in_question
    :return:
    """

    ccc_index = []
    ccc_tag = []
    ccc_in = []

    qqq_index = []
    qqq_tag = []
    qqq_in = []

    for t, c, q in zip(titles, contents, questions):
        if t != t:
            t_list = []
            t_tag = []
        else:
            if is_zh_or_en(t):
                t = t + '。'
                t_list, t_tag = split_word_zh(t, have_tag=True)
            else:
                t = t + '. '
                t_list, t_tag = split_word_en(t, have_tag=True)

        if c != c:
            c_list = []
            c_tag = []
        else:
            if is_zh_or_en(c):
                c_list, c_tag = split_word_zh(c, have_tag=True)
            else:
                c_list, c_tag = split_word_en(c, have_tag=True)

        c_list = t_list + c_list
        c_tag = t_tag + c_tag

        if not c_list:
            c_list = [' ']
            c_tag = ['x']

        if q != q:
            q_list = [' ']
            q_tag = ['x']
        else:
            if is_zh_or_en(q):
                q_list, q_tag = split_word_zh(q, have_tag=True)
            else:
                q_list, q_tag = split_word_en(q, have_tag=True)

        flag_c = []
        for cc in c_list:
            if cc in q_list:
                flag_c.append(1)
            else:
                flag_c.append(0)

        flag_q = []
        for qq in q_list:
            if qq in c_list:
                flag_q.append(1)
            else:
                flag_q.append(0)

        assert len(c_list) == len(c_tag) == len(flag_c)
        assert len(q_list) == len(q_list) == len(flag_q)

        ccc_index.append(c_list)
        ccc_tag.append(c_tag)
        ccc_in.append(flag_c)

        qqq_index.append(q_list)
        qqq_tag.append(q_tag)
        qqq_in.append(flag_q)

    return ccc_index, ccc_tag, ccc_in, qqq_index, qqq_tag, qqq_in


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


jieba.del_word('日电')
jieba.del_word('亿美元')
jieba.del_word('英国伦敦')


def split_word_zh(s, have_tag=False):
    """
     中文分词
    :param s: str
    :return: list
    """

    if have_tag is False:
        s = jieba.lcut(s, HMM=False)
        return s
    else:
        words, tags = list(zip(*posseg.lcut(s, HMM=False)))
        return list(words), list(tags)


def split_word_en(s, have_tag=False):
    """
    英文分词
    :param s: str
    :return:
    """
    if have_tag is False:
        s = nltk.word_tokenize(s)
        return s
    else:
        s = nltk.word_tokenize(s)
        _, tags = list(zip(*nltk.pos_tag(s)))
        return s, list(tags)


def is_zh_or_en(s):
    """
    判断一句话是否为中文
    :param s: str
    :return: True or False
    """
    flag = False
    for ss in list(s):
        if u'\u4e00' <= ss <= u'\u9fa5':
            flag = True
            break
    return flag


def words2index(words_list, vocab_path):
    """
    :param words_list: list of list
    :param vocab_path: file_path
    :return: list of list
    """
    with open(vocab_path, 'rb') as file:
        lang = pickle.load(file)
        w2i = lang['w2i']

    result = []
    for words in words_list:
        tmp = [w2i[word] if word in w2i else w2i['<unk>'] for word in words]
        result.append(tmp)

    return result


def tags2index(tags_list, tag_path):
    """
    :param tags_list:  list of list
    :param tag_path: file_path
    :return: list of list
    """
    with open(tag_path, 'rb') as file:
        lang = pickle.load(file)

    result = []
    for tags in tags_list:
        tmp = [lang[tag] if tag in lang else lang['<unk>'] for tag in tags]
        result.append(tmp)

    return result


def gen_str(titles, shorten_contents, questions, result_starts, result_ends, add_liangci=False):
    # if add_liangci:
    #     with open('data_gen/liangci_set.pkl', 'rb') as file:
    #         liangci_set = pickle.load(file)

    result = []
    ccc = 0
    cccc = 0
    for t, c, q, s, e in zip(titles, shorten_contents, questions, result_starts, result_ends):
        # 当问题等于标题时， 答案就是标题
        if q == t:
            result.append(t)
            continue

        # 当问题是标题类问题时， 答案就是标题
        if (t.strip() in title_question.question_titles) or (t.lower().strip() in title_question.question_titles):
            result.append(t)
            continue

        # 当标题为空时， 文本为空时， 答案为空
        if (t.strip() == '') and (c.strip() == ''):
            result.append('')
            continue

        # 如果问题为空，则答案也为空
        if q.strip() == '':
            result.append('')
            continue

        # 正常推断
        if t != t:
            t_list = []
        else:
            flag_t = is_zh_or_en(t)
            if flag_t:
                t_list = split_word_zh(t) + ['。']
            else:
                t_list = split_word_en(t) + ['.']

        if c != c:
            c_list = []
        else:
            flag_c = is_zh_or_en(c)
            if flag_c:
                c_list = split_word_zh(c)
            else:
                c_list = split_word_en(c)

        c_list = t_list + c_list
        r = c_list[s: e+1]

        if (s >= 0) and (s <= len(t_list)-1):
            if flag_t:
                r = ''.join(r)
            else:
                r = ' '.join(r)
        else:
            if flag_c:
                r = ''.join(r)
            else:
                r = ' '.join(r)

        # 为答案增加量词
        # if add_liangci:
        #     if len(r) >= 1 and r.isdigit() and len(c_list) > (e+1):
        #         r = r + c_list[e+1]
        #         ccc += 1

        # if add_liangci:
        #     if len(r) >= 1 and r[-1].isdigit() and len(c_list) > (e+1):
        #         ccc += 1
        #         word = c_list[e+1]
        #         if word in liangci_set:
        #             r = r + word
        #             cccc += 1
        #         else:
        #             for i in range(len(word)-1, 0, -1):
        #                 word_tmp = word[: i]
        #                 if word_tmp in liangci_set:
        #                     r = r + word_tmp
        #                     cccc += 1
        #                     break

        # 前后无空格
        r = r.strip()

        # 前后无标点
        if len(r) >= 1:
            drop_list_zh = ['。', '，', '、', '；', '：', '？', '！']
            drop_list_en = ['.', '?', '!', ';', ':', ',', '-', '...', '..', '....']
            drop_list = drop_list_zh + drop_list_en
            if r[-1] in drop_list:
                r = r[: -1].strip()
            if len(r) >= 1 and r[0] in drop_list:
                r = r[1:].strip()

        result.append(r)

    # print('add liangci %d/%d' % (cccc, ccc))

    return result
