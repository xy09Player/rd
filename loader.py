# encoding = utf-8
# author = xy

import numpy as np
import pandas as pd
import torch
from torch.utils import data
import utils


def load_w2v(embedding_path):
    """ load embedding vector """
    embedding_np = np.load(embedding_path)
    return embedding_np


def load_data(df_file, vocab_path, tag_path, c_max_len=500, q_max_len=150):
    """
    load data from .csv
    # 1. load
    # 2. index, tag(词性), 是否在答案中出现， 是否是标题
    # 3. padding
    return: content, question, answer_start, answer_end  (list)
    """

    # load
    df = pd.read_csv(df_file)
    titles = df['title'].values.tolist()
    contents = df['shorten_content'].values.tolist()
    questions = df['question'].values.tolist()

    if 'answer_start' in df:
        answer_start = df['answer_start'].values.tolist()
        answer_end = df['answer_end'].values.tolist()

    # words, flags, is_in
    c_index, c_tag, c_in, q_index, q_tag, q_in = utils.deal_data(titles, contents, questions)

    # words -> index
    c_index = utils.words2index(c_index, vocab_path)
    q_index = utils.words2index(q_index, vocab_path)

    # flags -> index
    c_tag = utils.tags2index(c_tag, tag_path)
    q_tag = utils.tags2index(q_tag, tag_path)

    # padding
    c_index = utils.pad(c_index, c_max_len)
    c_tag = utils.pad(c_tag, c_max_len)
    c_in = utils.pad(c_in, c_max_len)

    q_index = utils.pad(q_index, q_max_len)
    q_tag = utils.pad(q_tag, q_max_len)
    q_in = utils.pad(q_in, q_max_len)

    if 'answer_start' in df:
        return [c_index, c_tag, c_in, q_index, q_tag, q_in, answer_start, answer_end]
    else:
        return [c_index, c_tag, c_in, q_index, q_tag, q_in]


def build_loader(dataset, batch_size, shuffle, drop_last):
    """
    build data loader
    return: a instance of Dataloader
    """
    dataset = [torch.LongTensor(d) for d in dataset]
    dataset = data.TensorDataset(*dataset)
    data_iter = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter













