# coding = utf-8
# author = xy

import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder
import utils


class Model(nn.Module):
    """ bi-rdf for reading comprehension """

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

        # attention flow layer
        self.att_c = nn.Linear(self.hidden_size * 2, 1)
        self.att_q = nn.Linear(self.hidden_size * 2, 1)
        self.att_cq = nn.Linear(self.hidden_size * 2, 1)

        # modeling layer
        self.modeling_rnn = encoder.Rnn(
            mode=self.mode,
            input_size=self.hidden_size * 8,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            bidirectional=True,
            layer_num=2,
            is_bn=self.is_bn
        )

        # outputs
        self.p1 = nn.Linear(self.hidden_size * 10, 1)
        self.p2 = nn.Linear(self.hidden_size * 10, 1)

        self.rnn = encoder.Rnn(
            mode=self.mode,
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout_p=self.dropout_p,
            layer_num=1,
            is_bn=self.is_bn
        )

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, batch):
        """
        :param batch: [content, question, answer_start, answer_end]
        :return: ans_range (2, batch_size, content_len)
        """

        def att_flow_layer(c, c_mask, q, q_mask):
            """
            attention flow layer
            :param c: (c_len, batch_size, hidden_size*2)
            :param c_mask: (batch_size, c_len)
            :param q: (q_len, batch_size, hidden_size*2)
            :param q_mask: (batch_size, q_len)
            :return: g (c_len, batch_size, hidden_size*8)
            """
            c_len = c.size(0)
            q_len = q.size(0)
            batch_size = c.size(1)

            c = self.dropout(c)
            q = self.dropout(q)

            c = c.transpose(0, 1)
            q = q.transpose(0, 1)
            cq = c.unsqueeze(2).expand(batch_size, c_len, q_len, -1) * \
                 q.unsqueeze(1).expand(batch_size, c_len, q_len, -1)
            cq = self.att_cq(cq).squeeze(3)  # (batch_size, c_len, q_len)

            s = self.att_c(c).expand(batch_size, c_len, q_len) + \
                self.att_q(q).expand(batch_size, q_len, c_len).transpose(1, 2) + \
                cq

            # 除掉空位
            mask = c_mask.eq(0)
            mask = mask.unsqueeze(2).expand(batch_size, c_len, q_len)
            s.masked_fill_(mask, -1e30)  # 使用-float('inf')过滤，会和dropout冲突，有nan值。 使用小值过滤，不同batch_size输出结果不同

            mask = q_mask.eq(0)
            mask = mask.unsqueeze(1).expand(batch_size, c_len, q_len)
            s.masked_fill_(mask, -1e30)

            # c2q
            a = f.softmax(s, dim=2)
            c2q = torch.bmm(a, q)  # (batch_size, c_len, hidden_size*2)

            # q2c
            b = torch.max(s, dim=2)[0]
            b = f.softmax(b, dim=1)  # (batch_size, c_len)
            q2c = torch.bmm(b.unsqueeze(1), c).expand(batch_size, c_len, -1)  # (batch_size, c_len, hidden_size*2)

            x = torch.cat([c, c2q, c * c2q, c * q2c], dim=2)
            x = c_mask.unsqueeze(2) * x
            x = x.transpose(0, 1)

            return x

        def output_layer(g, m, c_mask):
            """
            output layer
            :param g: (c_len, batch_size, hidden_size*8)
            :param m: (c_len, batch_size, hidden_size*2)
            :param c_mask: (batch_size, c_len)
            :return: ans_range(2, batch_size, content_len)
            """
            gm = self.dropout(torch.cat([g, m], dim=2))
            p1 = self.p1(gm).squeeze(2).transpose(0, 1)  # (batch_size, c_len)

            m = self.rnn(m, c_mask)
            gm = self.dropout(torch.cat([g, m], dim=2))
            p2 = self.p2(gm).squeeze(2).transpose(0, 1)  # (batch_size, c_len)

            mask = c_mask.eq(0)
            p1.masked_fill_(mask, -float('inf'))
            p1 = f.softmax(p1, dim=1)
            p2.masked_fill_(mask, -float('inf'))
            p2 = f.softmax(p2, dim=1)

            result = torch.stack([p1, p2])

            # add 1e-6, and no gradient explosion
            new_mask = (c_mask - 1) * (-1e-30)
            result = result + new_mask.unsqueeze(0)

            return result

        content = batch[: 3]
        question = batch[3: 6]

        # mask
        content_mask = utils.get_mask(content[0])
        question_mask = utils.get_mask(question[0])

        # embedding
        content_vec = self.embedding(content)
        question_vec = self.embedding(question)

        # encode
        content_vec = self.encoder(content_vec, content_mask)
        question_vec = self.encoder(question_vec, question_mask)

        # attention flow layer
        g = att_flow_layer(content_vec, content_mask, question_vec, question_mask)

        # modeling layer
        m = self.modeling_rnn(g, content_mask)

        # outputs
        ans_range = output_layer(g, m, content_mask)

        return ans_range
