# coding = utf-8
# author = xy


import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import pointer
import utils


num_align_hops = 2
num_ptr_hops = 2


class Model(nn.Module):
    """ Reinforced Mnemonic Reader for Machine Comprehension 2017 """
    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        # embedding
        self.embedding = embedding.ExtendEmbedding(param['embedding'])

        # encoder
        input_size = self.embedding.embedding_dim
        self.encoder = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=True
        )

        # align
        self.aligner = nn.ModuleList([SeqToSeqAtten() for _ in range(num_align_hops)])
        self.aligner_sfu = nn.ModuleList([SFU(self.hidden_size*2, self.hidden_size*2*3, dropout_p=self.dropout_p)
                                          for _ in range(num_align_hops)])

        # self align
        self.self_aligner = nn.ModuleList([SelfSeqAtten() for _ in range(num_align_hops)])
        self.self_aligner_sfu = nn.ModuleList([SFU(self.hidden_size*2, self.hidden_size*2*3, dropout_p=self.dropout_p)
                                               for _ in range(num_align_hops)])

        # aggregation
        self.aggregation = nn.ModuleList([encoder.Rnn(
            mode=self.mode,
            input_size=self.hidden_size*2,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout_p=self.dropout_p,
            layer_num=1,
            is_bn=self.is_bn
        )
            for _ in range(num_align_hops)])

        # init zs
        self.init_state = pointer.AttentionPooling(
            input_size=self.hidden_size*2,
            output_size=self.hidden_size*2
        )

        # pointer
        self.ptr_net = nn.ModuleList([Pointer(self.hidden_size*2, self.hidden_size, self.dropout_p)
                                      for _ in range(num_ptr_hops)])

    def forward(self, batch):
        content = batch[: 3]
        question = batch[3: 6]

        # mask
        content_mask = utils.get_mask(content[0])
        question_mask = utils.get_mask(question[0])

        # embedding
        content_vec = self.embedding(content)
        question_vec = self.embedding(question)

        # encoder
        content_vec = self.encoder(content_vec, content_mask)
        question_vec = self.encoder(question_vec, question_mask)

        # aligner
        align_ct = content_vec
        for i in range(num_align_hops):
            qt_align_ct = self.aligner[i](align_ct, question_vec, question_mask)
            bar_ct = self.aligner_sfu[i](align_ct,
                                         torch.cat([qt_align_ct, align_ct*qt_align_ct, align_ct-qt_align_ct], dim=2))

            ct_align_ct = self.self_aligner[i](bar_ct, content_mask)
            hat_ct = self.self_aligner_sfu[i](bar_ct,
                                              torch.cat([ct_align_ct, bar_ct*ct_align_ct, bar_ct-ct_align_ct], dim=2))
            align_ct = self.aggregation[i](hat_ct, content_mask)

        # init state
        zs = self.init_state(question_vec, question_mask)

        # pointer
        for i in range(num_ptr_hops):
            ans_range, zs = self.ptr_net[i](align_ct, content_mask, zs)

        # add 1e-6
        content_mask = content_mask.float()
        new_mask = (content_mask - 1) * (-1e-30)
        ans_range = ans_range + new_mask.unsqueeze(0)

        return ans_range


class SeqToSeqAtten(nn.Module):
    def __init__(self):
        super(SeqToSeqAtten, self).__init__()

    def forward(self, content_vec, question_vec, question_mask):
        """
        :param content_vec: (c_len, batch_size, hidden_size)
        :param question_vec:
        :param question_mask:
        :return: (c_len, batch_size, hidden_size)
        """
        content_vec = content_vec.transpose(0, 1)  # (batch_size, c_len, hidden_size)
        question_vec = question_vec.transpose(0, 1)

        b = torch.bmm(content_vec, question_vec.transpose(1, 2))  # (batch_size, c_len, q_len)

        # mask
        mask = question_mask.eq(0).unsqueeze(1).expand(b.size())  # (batch_size, c_len, q_len)
        b.masked_fill_(mask, -float('inf'))

        b = f.softmax(b, dim=2)
        q = torch.bmm(b, question_vec)  # (batch_size, c_len, hidden_size)
        q = q.transpose(0, 1)  # (c_len, batch_size, hidden_size)

        return q


class SFU(nn.Module):
    def __init__(self, input_size, fusion_size, dropout_p):
        super(SFU, self).__init__()

        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs, fusions):
        """
        :param inputs:  (c_len, batch_size, input_size)
        :param fusions: (c_len, batch_size, input_size*3)
        :return: (c_len, batch_size, input_size)
        """

        m = torch.cat([inputs, fusions], dim=-1)
        m = self.dropout(m)
        r = torch.tanh(self.linear_r(m))
        g = torch.sigmoid(self.linear_g(m))
        o = g * r + (1-g) * inputs

        return o


class SelfSeqAtten(nn.Module):
    def __init__(self):
        super(SelfSeqAtten, self).__init__()

    def forward(self, h, h_mask):
        """
        :param h: (c_len, batch_size, input_size)
        :param h_mask: (batch_size, c_len)
        :return: (c_len, batch_size, input_size)
        """
        c_len = h.size(0)

        h = h.transpose(0, 1)  # (batch_size, c_len, input_size)
        alpha = torch.bmm(h, h.transpose(1, 2))  # (batch_size, c_len, c_len)

        # mask dialog
        mask = torch.eye(c_len, dtype=torch.uint8).cuda()
        mask = mask.unsqueeze(0)
        alpha.masked_fill_(mask, 0.0)

        # mask inf
        mask = h_mask.eq(0).unsqueeze(1).expand(alpha.size())  # (batch_size, c_len, c_len)
        alpha.masked_fill_(mask, -float('inf'))

        alpha = f.softmax(alpha, dim=2)
        o = torch.bmm(alpha, h)  # (batch_size, c_len, input_size)
        o = o.transpose(0, 1)

        return o


class Pointer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(Pointer, self).__init__()

        self.start_net = FN(input_size*3, hidden_size, dropout_p)
        self.sfu_1 = SFU(input_size, input_size, dropout_p)

        self.end_net = FN(input_size*3, hidden_size, dropout_p)
        self.sfu_2 = SFU(input_size, input_size, dropout_p)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, h, h_mask, zs):
        """
        :param h: (c_len, batch_size, input_size)
        :param h_mask: (batch_size, c_len)
        :param zs: (batch_size, input_size)
        :return: o(2, batch_size, c_len), zs(batch_size, input_size)
        """
        h = self.dropout(h)

        # start position
        zs_tmp = zs.unsqueeze(0).expand(h.size())  # (c_len, batch_size, input_size)
        x = torch.cat([h, zs_tmp, h*zs_tmp], dim=2)  # (c_len, batch_size, input_size*3)
        start_p = self.start_net(x, h_mask)  # (batch_size, c_len)

        # end position
        us = torch.bmm(start_p.unsqueeze(1), h.transpose(0, 1)).squeeze(1)  # (batch_size, input_size)
        ze = self.sfu_1(zs, us)  # (batch_size, input_size)

        ze_tmp = ze.unsqueeze(0).expand(h.size())
        x = torch.cat([h, ze_tmp, h*ze_tmp], dim=2)
        end_p = self.end_net(x, h_mask)  # (batch_size, c_len)

        # update
        ue = torch.bmm(end_p.unsqueeze(1), h.transpose(0, 1)).squeeze(1)
        zs = self.sfu_2(ze, ue)

        # output
        o = torch.stack([start_p, end_p])

        return o, zs


class FN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(FN, self).__init__()

        self.linear_h = nn.Linear(input_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, x_mask):
        """
        :param x: (c_len, batch_size, input_size)
        :param x_mask: (batch_size, c_len)
        :return: (batch_size, c_len)
        """
        h = f.relu(self.linear_h(x))
        h = self.dropout(h)
        o = self.linear_o(h).squeeze(2)
        o = o.transpose(0, 1)  # (batch_size, c_len)

        # mask
        mask = x_mask.eq(0)
        o.masked_fill_(mask, -float('inf'))
        o = f.softmax(o, dim=1)

        return o
