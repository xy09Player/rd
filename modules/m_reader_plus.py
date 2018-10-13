# coding = utf-8
# author = xy


import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder
import utils


class Model(nn.Module):
    """ Reinforced Mnemonic Reader for Machine Reading Comprehension 2018 """
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
        input_size = self.hidden_size * 2
        self.align_1 = Aligner(input_size, self.dropout_p, self.mode, self.is_bn, False)
        self.align_2 = Aligner(input_size, self.dropout_p, self.mode, self.is_bn, False)
        self.align_3 = Aligner(input_size, self.dropout_p, self.mode, self.is_bn, True)

        # pointer
        self.pointer = Pointer(input_size, self.dropout_p)

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
        R1, Z1, E1, B1 = self.align_1(content_vec, content_mask, question_vec, question_mask)
        R2, Z2, E2, B2 = self.align_2(R1, content_mask, question_vec, question_mask, E1, B1)
        R3, _, _, _ = self.align_3(R2, content_mask, question_vec, question_mask, E2, B2, Z1, Z2)

        # pointer
        out = self.pointer(R3, content_mask, question_vec, question_mask)

        return out


class Aligner(nn.Module):
    def __init__(self, input_size, dropout_p, mode, is_bn, use_rnn):
        super(Aligner, self).__init__()

        self.inter_align = InterAlign(input_size, dropout_p)
        self.self_align = SelfAlign(input_size, dropout_p)
        self.aggregation = EviCollection(mode, input_size, dropout_p, is_bn)

        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn = encoder.Rnn(
                mode=mode,
                input_size=input_size*3,
                hidden_size=input_size//2,
                dropout_p=dropout_p,
                bidirectional=True,
                layer_num=1,
                is_bn=is_bn
            )

    def forward(self, U, U_mask, V, V_mask, E=None, B=None, Z1=None, Z2=None):
        """
        :param U: (c_len, batch_size, input_size)
        :param V: (q_len, batch_size, input_size)
        :param E: (batch_size, c_len, q_len)
        :param B:
        :param Z1: (c_len, batch_size, input_size)
        :param Z2:
        :return:R,Z,E,B
        """
        h, E = self.inter_align(U, U_mask, V, V_mask, E, B)
        Z, B = self.self_align(h, U_mask, B)
        if self.use_rnn:
            z = torch.cat([Z1, Z2, Z], dim=2)
            R = self.rnn(z, U_mask)
        else:
            R = self.aggregation(Z, U_mask)

        return R, Z, E, B


class InterAlign(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(InterAlign, self).__init__()

        self.Wu = nn.Linear(input_size, input_size)
        self.Wv = nn.Linear(input_size, input_size)
        self.gamma = nn.Parameter(torch.tensor(3.0))
        self.sfu = SFU(input_size, dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, content_vec, content_mask, question_vec, question_mask, E_t=None, B_t=None):
        """
        :param content_vec: (c_len, batch_size, input_size)
        :param content_mask: (batch_size, c_len)
        :param question_vec: (q_len, batch_size, input_size)
        :param question_mask: (batch_size, q_len)
        :param E_t: (batch_size, c_len, q_len)  or None
        :param B_t: (batch_size, c_len, c_len) or None
        :return: h(c_len, batch_size, input_size), E_t(batch_size, c_len, q_len)
        """

        content_vec = self.dropout(content_vec)
        question_vec = self.dropout(question_vec)

        # E_t
        content_vec_tmp = f.relu(self.Wu(content_vec)).transpose(0, 1)  # (batch_size, c_len, input_size)
        question_vec_tmp = f.relu(self.Wv(question_vec)).transpose(0, 1)  # (batch_size, q_len, input_size)
        E_0 = torch.bmm(content_vec_tmp, question_vec_tmp.transpose(1, 2))  # (batch_size, c_len, q_len)

        if E_t is not None:
            E_t_mask = content_mask.eq(0).unsqueeze(2).expand(E_t.size())
            E_t = E_t.masked_fill(E_t_mask, -float('inf'))
            E_t = f.softmax(E_t, dim=1)  # (batch_size, c_len, q_len)

            B_t_mask = content_mask.eq(0).unsqueeze(1).expand(B_t.size())
            B_t = B_t.masked_fill(B_t_mask, -float('inf'))
            B_t = f.softmax(B_t, dim=2)  # (batch_size, c_len, c_len)

            E_1 = torch.bmm(B_t, E_t)
            E_t = E_0 + self.gamma * E_1  # (batch_size, c_len, q_len)
        else:
            E_t = E_0

        # V_bar
        mask = question_mask.eq(0).unsqueeze(1).expand(E_t.size())
        E_tt = E_t.masked_fill(mask, -float('inf'))
        E_tt = f.softmax(E_tt, dim=2)  # (batch_size, c_len, q_len)
        question_vec_tmp = torch.bmm(E_tt, question_vec.transpose(0, 1))  # (batch_size, c_len, input_size)
        question_vec_tmp = question_vec_tmp.transpose(0, 1)  # (c_len, batch_size, input_size)

        # fusion
        h = self.sfu(content_vec, question_vec_tmp)  # (c_len, batch_size, input_size)

        return h, E_t


class SelfAlign(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(SelfAlign, self).__init__()

        self.W1 = nn.Linear(input_size, input_size)
        self.W2 = nn.Linear(input_size, input_size)
        self.gamma = nn.Parameter(torch.tensor(3.0))
        self.sfu = SFU(input_size, dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, h, h_mask, B_t=None):
        """
        :param h: (c_len, batch_size, input_size)
        :param h_mask: (batch_size, c_len)
        :param B_t: (batch_size, c_len_1, c_len_2) or None
        :return: h(c_len, batch_size, input_size), B_t(batch_size, c_len, c_len)
        """

        h = self.dropout(h)  # (c_len_2, batch_size, input_size)

        # B_t
        h_tmp_1 = f.relu(self.W1(h)).transpose(0, 1)  # (batch_size, c_len_1, input_size)
        h_tmp_2 = f.relu(self.W2(h)).transpose(0, 1)  # (batch_size, c_len_2, input_size)
        B_0 = torch.bmm(h_tmp_1, h_tmp_2.transpose(1, 2))  # (batch_size, c_len_1, c_len_2)

        if B_t is not None:
            mask_1 = h_mask.eq(0).unsqueeze(2).expand(B_t.size())
            B_t_1 = B_t.masked_fill(mask_1, -float('inf'))
            B_t_1 = f.softmax(B_t_1, dim=1)  # (batch_size, c_len_1, c_len_2)

            mask_2 = h_mask.eq(0).unsqueeze(1).expand(B_t.size())
            B_t_2 = B_t.masked_fill(mask_2, -float('inf'))
            B_t_2 = f.softmax(B_t_2, dim=2)  # (batch_size, c_len_1, c_len_1)

            B_1 = torch.bmm(B_t_2, B_t_1)  # (batch_size, c_len_1, c_len_2)
            B_t = B_0 + self.gamma * B_1

        else:
            B_t = B_0

        # make dialog
        mask = torch.eye(h.size(0), dtype=torch.uint8).cuda()
        mask = mask.unsqueeze(0)
        B_t.masked_fill_(mask, 0.0)

        # h_bar
        mask = h_mask.eq(0).unsqueeze(1).expand(B_t.size())
        B_tt = B_t.masked_fill(mask, -float('inf'))
        B_tt = f.softmax(B_tt, dim=2)
        h_tmp = torch.bmm(B_tt, h.transpose(0, 1))  # (batch_size, c_len_1, input_size)
        h_tmp = h_tmp.transpose(0, 1)  # (c_len_1, batch_size, input_size)

        # fusion
        h = self.sfu(h, h_tmp)  # (c_len_1, batch_size, input_size)

        return h, B_t


class EviCollection(nn.Module):
    def __init__(self, mode, input_size, dropout_p, is_bn):
        super(EviCollection, self).__init__()

        self.rnn = encoder.Rnn(
            mode=mode,
            input_size=input_size,
            hidden_size=input_size//2,
            dropout_p=dropout_p,
            bidirectional=True,
            layer_num=1,
            is_bn=is_bn
        )

    def forward(self, z, z_mask):
        """
        :param z: (c_len, batch_size, input_size)
        :return: (c_len, batch_size, input_size)
        """

        o = self.rnn(z, z_mask)
        return o


class SFU(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(SFU, self).__init__()
        self.Wr = nn.Linear(input_size*4, input_size, bias=False)
        self.Wg = nn.Linear(input_size*4, input_size, bias=False)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs, fusion):
        """
        :param inputs: (.., input_size)
        :param fusion: (.., fusion_size)
        :return: (.., input_size)
        """
        m = torch.cat([inputs, fusion, inputs*fusion, inputs-fusion], dim=-1)
        m = self.dropout(m)
        x = f.relu(self.Wr(m))
        g = torch.sigmoid(self.Wg(m))
        o = x*g + (1-g)*inputs

        return o


class Pointer(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(Pointer, self).__init__()

        self.wq = nn.Linear(input_size, 1)

        self.W1 = nn.Linear(input_size*4, input_size)
        self.w1 = nn.Linear(input_size, 1)

        self.W2 = nn.Linear(input_size*4, input_size)
        self.w2 = nn.Linear(input_size, 1)

        self.sfu = SFU(input_size, dropout_p)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, R, R_mask, V, V_mask):
        """
        :param R: (c_len, batch_size, input_size)
        :param R_mask:
        :param V: (q_len, batch_size, input_size)
        :param V_mask:
        :return: (2, batch_size, c_len)
        """
        R = R.transpose(0, 1)  # (batch_size, c_len, input_size)

        # s
        s = self.wq(V).squeeze(2).transpose(0, 1)  # (batch_size, q_len)
        mask = V_mask.eq(0)
        s.masked_fill_(mask, -float('inf'))
        s = f.softmax(s, dim=1)
        s = torch.bmm(s.unsqueeze(1), V.transpose(0, 1))  # (batch_size, 1, input_size)
        s = s.expand(R.size())  # (batch_size, c_len, input_size)

        # p_start
        m = torch.cat([R, s, R*s, R-s], dim=2)  # (bach_size, c_len, input_size*4)
        m = self.dropout(m)
        m = torch.tanh(self.W1(m))  # (batch_size, c_len, input_size)
        p_start = self.w1(m).squeeze(2)  # (batch_size, c_len)
        mask = R_mask.eq(0)
        p_start.masked_fill_(mask, -float('inf'))
        p_start = f.softmax(p_start, dim=1)  # (batch_size, c_len)

        # update s
        l = torch.bmm(p_start.unsqueeze(1), R).expand(s.size())
        s = self.sfu(s, l)

        # end_start
        m = torch.cat([R, s, R*s, R-s], dim=2)
        m = self.dropout(m)
        m = torch.tanh(self.W2(m))
        p_end = self.w2(m).squeeze(2)
        p_end.masked_fill_(mask, -float('inf'))
        p_end = f.softmax(p_end, dim=1)  # (batch_size, c_len)

        result = torch.stack([p_start, p_end])

        # add 1e-6
        result = result + ((1 - R_mask.float()) * 1e-30).unsqueeze(0)

        return result
