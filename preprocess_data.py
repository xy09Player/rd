# coding = utf-8
# author = xy

import json
from rouge import Rouge
import pandas as pd
import numpy as np
import copy
import os
from data_pre import title_question
from data_pre import clean_data
import utils
import pickle
import time
import sys
import gensim
from sklearn import model_selection
from config import config_base

config = config_base.config


# convert .json to .pandas
# return: df
def organize_data(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
        result = []
        for dc in data:
            temp = [dc['article_id'], dc['article_type'], dc['article_title'], dc['article_content']]
            for items in dc['questions']:
                r = copy.deepcopy(temp)
                r = r + list(items.values())
                result.append(r)
        if 'answer' in data[0]['questions'][0]:
            columns = ['article_id', 'article_type', 'article_title', 'article_content', 'question_id',
                       'article_question', 'article_answer', 'question_type']
        else:
            columns = ['article_id', 'article_type', 'article_title', 'article_content', 'question_id',
                       'article_question']
        df = pd.DataFrame(data=result, columns=columns)

        return df


# 1. 除掉多余空格、删除'\u3000'
# 2. 繁简体转换
# 3. 删除答案中，不好的结尾符
def deal_data_for_train(df):

    # 1, 2
    df['title'] = clean_data.deal_data(df['article_title'].values)
    df['content'] = clean_data.deal_data(df['article_content'].values)
    df['question'] = clean_data.deal_data(df['article_question'].values)

    # 3
    if 'article_answer' in df:
        df['answer'] = clean_data.deal_data(df['article_answer'].values)

        # 除掉句首句尾标点，及空格
        answers = df[df['answer'] != '']['answer'].values
        drop_list_zh = ['。', '，', '、', '；', '：', '？', '！']
        drop_list_en = ['.', '?', '!', ';', ':', ',', '-', '...', '..', '....']
        drop_list = drop_list_zh + drop_list_en
        answers = [answer[:-1].strip() if answer[-1] in drop_list else answer for answer in answers]
        answers = [answer[1:].strip() if answer[0] in drop_list else answer for answer in answers]

        df.loc[df['answer'] != '', 'answer'] = answers

    return df


def deal_data_for_test(df):
    # 主要是除掉 前后空格
    titles = df['article_title'].values
    titles = [t.strip() for t in titles]
    df['title'] = titles

    contents = df['article_content'].values
    contents = [c.strip() for c in contents]
    df['content'] = contents

    questions = df['article_question'].values
    questions = [q.strip() for q in questions]
    df['question'] = questions

    return df


# shorten content
def shorten_content_all(df, max_len):
    """
    :param df:
    :param max_len:
    :return: df
    """
    sys.setrecursionlimit(1000000)
    rouge = Rouge(metrics=['rouge-l'])

    def match(title, content, question, max_len):

        title_is_zh = utils.is_zh_or_en(title)
        content_is_zh = utils.is_zh_or_en(content)

        if title_is_zh:
            title_list = utils.split_word_zh(title)
        else:
            title_list = utils.split_word_en(title)

        def count(flag, content_list):
            """ 查数 """
            number = 0
            for i in range(len(flag)):
                if flag[i] != 0:
                    number += len(content_list[i])+1
            return number

        # 过滤
        if content_is_zh:
            title_number = len(title_list)
            content_number = len(utils.split_word_zh(content))
            if (title_number + content_number + 1) <= max_len:
                return content

        if content_is_zh:
            if '。' not in content:
                c_list = utils.split_word_zh(content)
                c_list = c_list[: config.max_len-len(title_list)-1]
                return ''.join(c_list)

            content_list = content.split('。')
            temp = []
            for c in content_list:
                if c not in ['', ' ', '  ']:
                    temp.append(c)
            content_list = temp
            content_list = [utils.split_word_zh(c) for c in content_list]
            content_list = [title_list] + content_list
            content_len = len(content_list)

            if utils.is_zh_or_en(question):
                question_list = utils.split_word_zh(question)
            else:
                question_list = utils.split_word_en(question)
            question_str = ' '.join(question_list)

            # 相似性得分: rouge-l
            scores = []
            for c in content_list:
                if ''.join(c) in question:
                    scores.append(-5)
                    continue
                c_str = ' '.join(c)
                score = rouge.get_scores(c_str, question_str, avg=True)['rouge-l']['r']
                scores.append(score)

            # 标记类型
            flag = np.zeros(content_len)
            # title_number = len(utils.split_word_zh(title))
            # max_len = max_len - title_number

            # 核心句:
            max_score = max(scores)
            for i in range(content_len):
                if scores[i] == max_score:
                    flag[i] = -1

            # 核心句下一句
            for i in range(content_len):
                if (flag[i] == -1) and (i+1 < content_len) and (flag[i+1] == 0):
                    flag[i+1] = -2

            # 最后一句
            if flag[-1] == 0:
                flag[-1] = -3

            # 第一句
            if flag[1] == 0:
                flag[1] = -4

            # 蕴含句（上+中+下）
            for i in range(content_len):
                if scores[i] == -5:
                    if (i-1 >= 0) and (flag[i-1] == 0):
                        flag[i-1] = -5
                    if flag[i] == 0:
                        flag[i] = -5
                    if (i+1 < content_len) and (flag[i+1] == 0):
                        flag[i+1] = -5

            # 核心句下下句
            for i in range(content_len):
                if (flag[i] == -1) and (i+2 < content_len) and (flag[i+2] == 0):
                    flag[i+2] = -6

            # 核心句上一句
            for i in range(content_len):
                if (flag[i] == -1) and (i-1 >= 0) and (flag[i-1] == 0):
                    flag[i-1] = -7

            # 核心句下下下句
            for i in range(content_len):
                if (flag[i] == -1) and (i+3 < content_len) and (flag[i+3] == 0):
                    flag[i+3] = -8

            # 核心句上上句
            for i in range(content_len):
                if (flag[i] == -1) and (i-2 >= 0) and (flag[i-2] == 0):
                    flag[i-2] = -9

            # 倒数第二句
            if(len(flag) >= 3) and (flag[-2] == 0):
                flag[-2] = -10

            # 第二句
            if (len(flag) >= 3) and (flag[2] == 0):
                flag[2] = -11

            flag[0] = 0
            number = count(flag, content_list)
            max_len = max_len - len(title_list)
            result = []
            if number <= max_len:
                for i in range(content_len):
                    if flag[i] != 0:
                        result.append(''.join(content_list[i]))
            else:
                flag_copy = np.zeros(content_len)
                c_count = 0
                xxx = True
                for i in range(-1, -12, -1):
                    for j in range(len(flag)):
                        if flag[j] == i:
                            c_count = c_count + len(content_list[j]) + 1
                            if c_count <= max_len:
                                flag_copy[j] = -1
                            else:
                                xxx = False
                                break
                    if xxx is False:
                        break

                for i in range(content_len):
                    if flag_copy[i] == -1:
                        result.append(''.join(content_list[i]))

                if len(result) == 0:
                    for j in range(len(flag)):
                        if flag[j] == -1:
                            result = [''.join(content_list[j][: max_len-1])]
                            break

            # 过滤重复
            temp = []
            for r in result:
                if r not in temp:
                    temp.append(r)
            result = temp

            return '。'.join(result)

        else:
            www = content
            words = utils.split_word_en(www)
            if (len(words) + len(title_list) + 1) <= config.max_len:
                return content
            else:
                index = 0
                for i in words[: config.max_len-len(title_list)-1]:
                    index = index + len(i)

                while content[index] != ' ':
                    index = index + 1

                return content[: index]

    titles = df['title'].values
    contents = df['content'].values
    questions = df['question'].values

    shorten_content = [match(t, c, q, max_len) for t, c, q in zip(titles, contents, questions)]
    df['shorten_content'] = shorten_content

    # 评估数据集构建效果
    if 'answer' in df:

        answers = df['answer'].values

        is_in = [True if (a in c) or (a in t) else False for c, t, a in zip(contents, titles, answers)]
        r1 = sum(is_in)/len(df)
        print('答案存在比例：%.4f' % r1)

        is_in = [True if (a in t) or (a in m) else False for t, m, a in zip(titles, shorten_content, answers)]
        df['is_in'] = is_in
        r2 = sum(is_in)/len(df)
        print('截取比例：%.4f' % r2)

        print('截取准确率：%.4f' % (r2/r1))

    merge_len = []
    for t, s in zip(titles, shorten_content):
        if utils.is_zh_or_en(t):
            len_t = len(utils.split_word_zh(t))
        else:
            len_t = len(utils.split_word_en(t))

        if utils.is_zh_or_en(s):
            len_s = len(utils.split_word_zh(s))
        else:
            len_s = len(utils.split_word_en(s))
        len_m = len_t + len_s + 1
        merge_len.append(len_m)

    df['len'] = merge_len
    print('max length: %d' % max(merge_len))
    print('min length: %d' % min(merge_len))
    print('mean length:%d' % df['len'].mean())
    print('median length:%d' % df['len'].median())

    return df


# build answer_range
def build_answer_range(df):
    sys.setrecursionlimit(1000000)
    rouge = Rouge(metrics=['rouge-l'])

    def match(title, shorten_content, answer, question):
        if utils.is_zh_or_en(title):
            title_list = utils.split_word_zh(title) + ['。']
        else:
            title_list = utils.split_word_en(title) + ['.']

        if utils.is_zh_or_en(shorten_content):
            content_list = utils.split_word_zh(shorten_content)
        else:
            content_list = utils.split_word_en(shorten_content)

        merge_list = title_list + content_list
        merge_len = len(merge_list)

        if utils.is_zh_or_en(question):
            answer_list = utils.split_word_zh(answer)
            question_list = utils.split_word_zh(question)
            question_str = ' '.join(question_list)
        else:
            answer_list = utils.split_word_en(answer)
            question_list = utils.split_word_en(question)
            question_str = ' '.join(question_list)

        answer_len = len(answer_list)

        start = []
        end = []
        if answer == '':
            return -1, -1
        for i in range(0, merge_len-answer_len+1):
            if merge_list[i: i+answer_len] == answer_list:
                start.append(i)
                end.append(i+answer_len-1)
        if len(start) == 0:
            return -1, -1
        elif len(start) == 1:
            return start[0], end[0]
        else:
            scores = []
            # 前后扩展5个词
            for s, e in zip(start, end):
                s = max(s-5, 0)
                answer_can = ' '.join(merge_list[s: e+5])
                score = rouge.get_scores(answer_can, question_str, avg=True)['rouge-l']['r']
                scores.append(score)
            max_idx = np.argmax(scores)
            return start[max_idx], end[max_idx]

    titles = df[df['is_in']]['title'].values
    shortens = df[df['is_in']]['shorten_content'].values
    answers = df[df['is_in']]['answer'].values
    questions = df[df['is_in']]['question'].values
    answer_range = [match(t, s, a, q) for t, s, a, q in zip(titles, shortens, answers, questions)]

    start, end = list(zip(*answer_range))
    df.loc[df['is_in'], 'answer_start'] = start
    df.loc[df['is_in'], 'answer_end'] = end

    merge_len = len(titles)
    right_all_len = (df['answer_end'] >= 0).sum()
    wrong_split_len = (df['answer_end'] == -1).sum()
    print('answer generation accuracy(all): %.4f' % (right_all_len/merge_len))
    print('wrong split: %.4f' % (wrong_split_len/merge_len))

    return df


# 将不适宜训练的数据，for_train = False
def select_data(df):
    questions = df['question'].values
    titles = df['title'].values
    answers = df['answer'].values

    # 标题和问题重复的数据
    flag_1 = []
    for q, t in zip(questions, titles):
        if q == t:
            flag_1.append(False)
        else:
            flag_1.append(True)
    print('标题和问题重复的数据：%.4f' % (1-sum(flag_1)/len(flag_1)))

    # 问题和答案重复的数据
    flag_2 = []
    for q, a in zip(questions, answers):
        if q == a:
            flag_2.append(False)
        else:
            flag_2.append(True)
    print('问题和答案重复的数据:%.4f' % (1-sum(flag_2)/len(flag_2)))

    # 如果问题是“标题是什么”时，此时答案和标题不同的数据
    flag_3 = []
    for q, t, a in zip(questions, titles, answers):
        if (q in title_question.question_titles) and (t != a):
            flag_3.append(False)
        else:
            flag_3.append(True)
    print('当问题是标题类问题时，答案与标题不同的数据:%.4f' % (1-sum(flag_3)/len(flag_3)))

    flag = []
    for i, j, k in zip(flag_1, flag_2, flag_3):
        if (i is True) and (j is True) and (k is True):
            flag.append(True)
        else:
            flag.append(False)
    print('select data: %.4f' % (sum(flag)/len(flag)))

    df['for_train'] = flag
    return df


# build train, val, test dataset
def split_dataset(df):
    # deal data: 能找到答案
    all_data = len(df)
    print('all data size:%d' % all_data)
    # split train, val dataset
    train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=0)
    test_df = val_df.copy()

    # deal train, val data
    train_len = len(train_df)
    train_df = train_df[train_df['answer_start'] > -1]
    train_df = train_df[train_df['answer_end'] > -1]
    train_df = train_df[train_df['for_train']]
    train_df = train_df[['question', 'title', 'shorten_content', 'answer_start', 'answer_end']]
    print('train size:%d, shorten train size:%d' % (train_len, len(train_df)))

    # deal val data
    val_len = len(val_df)
    val_df = val_df[val_df['answer_start'] > -1]
    val_df = val_df[val_df['answer_end'] > -1]
    val_df = val_df[val_df['for_train']]
    val_df = val_df[['question', 'title', 'shorten_content', 'answer_start', 'answer_end']]
    print('val size:%d, shorten val size:%d' % (val_len, len(val_df)))

    # deal test data
    print('fake test data size:%d' % len(val_df))
    test_df = test_df

    return train_df, val_df, test_df


def build_vocab_embedding(list_df, vocab_path, embedding_in_zh, embedding_in_en, embedding_out):
    data = []
    for df in list_df:
        if 'answer' in df:
            data += df[['title', 'content', 'question', 'answer']].values.flatten().tolist()
        else:
            data += df[['title', 'content', 'question']].values.flatten().tolist()

    vocab = set()
    for d in data:
        if utils.is_zh_or_en(d):
            d_list = utils.split_word_zh(d)
        else:
            d_list = utils.split_word_en(d)
        for dd in d_list:
            vocab.add(dd)
    print('data, word_num:%d' % len(vocab))

    # zh
    model_zh = gensim.models.KeyedVectors.load_word2vec_format(embedding_in_zh)
    # en
    model_en = gensim.models.KeyedVectors.load_word2vec_format(embedding_in_en)

    tmp = set()
    for word in vocab:
        if word in model_zh or word in model_en:
            tmp.add(word)
    print('word_nums in pre-embedding:%d/%d, radio:%.4f' % (len(tmp), len(vocab), len(tmp)/len(vocab)))

    w2i = {'<pad>': 0}
    i2w = {0: '<pad>'}
    c = 1
    embedding = np.zeros([len(tmp) + 3, model_zh.vector_size])
    for word in tmp:
        w2i[word] = c
        i2w[c] = word
        if word in model_zh:
            embedding[c] = model_zh[word]
        elif word in model_en:
            embedding[c] = model_en[word]
        c += 1
    w2i['<unk>'] = len(tmp) + 1
    i2w[len(tmp)+1] = '<unk>'
    w2i[' '] = len(tmp) + 2
    i2w[len(tmp)+2] = ' '
    lang = {'w2i': w2i, 'i2w': i2w}
    print('vocab size:%d' % (c+2))
    print('embedding size:', embedding.shape)

    # save
    with open(vocab_path, 'wb') as file:
        pickle.dump(lang, file)
    np.save(embedding_out, embedding)


# 生成 词性-index 表
def gen_tag_index(df):
    df = df[['title', 'content', 'question']]
    data = df.values.flatten().tolist()
    tag2i = {'<pad>': 0, '<unk>': 1}
    cc = 2
    for d in data:
        if utils.is_zh_or_en(d):
            _, tags = utils.split_word_zh(d, have_tag=True)
        else:
            _, tags = utils.split_word_en(d, have_tag=True)

        for t in tags:
            if t not in tag2i:
                tag2i[t] = cc
                cc += 1

    with open(config.tag_path, 'wb') as file:
        pickle.dump(tag2i, file)
    print('word flag num:%d' % len(tag2i))  # 98个


def gen_pre_file_for_train():
    if os.path.isfile(config.train_vocab_path) is False:
        time0 = time.time()
        print('gen train prepared file...')

        # 组织数据 json -> df
        df = organize_data(config.train_data)

        # 数据预处理
        df = deal_data_for_train(df)

        # vocab, embedding
        build_vocab_embedding(
            list_df=[df],
            vocab_path=config.train_vocab_path,
            embedding_in_zh=config.pre_embedding_zh,
            embedding_in_en=config.pre_embedding_en,
            embedding_out=config.train_embedding
        )

        # 生成词性表
        gen_tag_index(df)

        print('gen train prepared file, time；%d' % (time.time()-time0))


def gen_pre_file_for_test():
    if os.path.isfile(config.test_vocab_path) is False:
        time0 = time.time()
        print('gen test prepared file...')
        # 组织数据 json -> df
        df = organize_data(config.test_data)
        # 数据预处理
        df = deal_data_for_test(df)
        # vocab, embedding
        build_vocab_embedding(
            list_df=[df],
            vocab_path=config.test_vocab_path,
            embedding_in_zh=config.pre_embedding_zh,
            embedding_in_en=config.pre_embedding_en,
            embedding_out=config.test_embedding
        )
        print('gen test prepared file, time:%d' % (time.time()-time0))


def gen_train_datafile():
    if os.path.isfile(config.train_df) is False:
        time0 = time.time()
        print('gen train data...')

        # read .json
        df = organize_data(config.train_data)
        # 预处理数据
        df = deal_data_for_train(df)
        # shorten content
        df = shorten_content_all(df, config.max_len)
        # answer_range
        df = build_answer_range(df)
        # 确定脏数据
        df = select_data(df)
        # split train, val
        df_train, df_val, df_test = split_dataset(df)
        # to .csv
        df_train.to_csv(config.train_df, index=False)
        df_val.to_csv(config.val_df, index=False)
        df_test.to_csv(config.test_val_df, index=False)

        print('gen train data time:%d' % (time.time()-time0))


def gen_test_datafile():
    time0 = time.time()
    print('gen test data...')

    # read .json
    df = organize_data(config.test_data)
    # 预处理数据
    df = deal_data_for_test(df)
    # shorten content
    df = shorten_content_all(df, config.max_len)
    # to .csv
    df.to_csv(config.test_df, index=False)

    print('test data size:%d' % len(df))
    print('gen test data time:%d' % (time.time()-time0))

if __name__ == '__main__':
    pass





















