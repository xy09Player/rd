# coding = utf-8
# author = xy

from torch import nn
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import match_rnn
from modules.layers import self_match_attention
from modules.layers import pointer
import utils


class Model(nn.Module):
    """ r-net for machine comprehension """
    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

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

        # match rnn
        input_size = self.hidden_size * 2
        self.match_rnn = match_rnn.MatchRNN(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            gated_attention=True,
            is_bn=self.is_bn
        )

        # self matching attention
        input_size = self.hidden_size * 2
        self.self_match_attention = self_match_attention.SelfAttention(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            gated_attention=True,
            is_bn=self.is_bn
        )

        # addition_rnn
        input_size = self.hidden_size * 2
        self.addition_rnn = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout_p=self.dropout_p,
            layer_num=1,
            is_bn=self.is_bn
        )

        # init state of pointer
        self.init_state = pointer.AttentionPooling(
            input_size=self.hidden_size*2,
            output_size=self.hidden_size
        )

        # pointer
        input_size = self.hidden_size * 2
        self.pointer_net = pointer.BoundaryPointer(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            bidirectional=True,
            is_bn=self.is_bn
        )

    def forward(self, batch):
        """
        :param batch: [content, question, answer_start, answer_end]
        :return: ans_range(2, batch_size, content_len)
        """
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

        # match rnn
        hr = self.match_rnn(content_vec, content_mask, question_vec, question_mask)

        # self matching attention
        hr = self.self_match_attention(hr, content_mask)

        # aggregation
        hr = self.addition_rnn(hr, content_mask)

        # init state of pointer
        init_state = self.init_state(question_vec, question_mask)

        # pointer
        ans_range = self.pointer_net(hr, content_mask, init_state)

        return ans_range



