# coding = utf-8
# author = xy


import torch
import utils
from torch.nn.modules import loss
import torch.nn.functional as f


class MyNLLLoss(loss._Loss):
    """ MLE 最大似然估计 """
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, outputs, batch):
        """
        :param outputs: tensor (2, batch_size, content_seq_len)
        :param batch: tensor
        :return:loss
        """
        y_start = batch[-2]
        y_end = batch[-1]
        outputs = torch.log(outputs)
        start_loss = f.nll_loss(outputs[0], y_start)
        end_loss = f.nll_loss(outputs[1], y_end)
        return start_loss + end_loss


class RougeLoss(loss._Loss):
    """ MRT 最小风险 """
    def __init__(self, lam):
        super(RougeLoss, self).__init__()

        self.lam = lam

        self.mle = MyNLLLoss()

    def forward(self, outputs, batch):
        """
        :param outputs: tensor (2, batch_size, content_seq_len)
        :param batch: tensor (content, question, start, end)
        :return: loss
        """
        # mle
        loss_mle = self.mle(outputs, batch)

        # mrt
        batch_size = outputs.size(1)
        c_len = outputs.size(2)
        start_y = batch[-2].view(-1, 1).float()  # (batch_size, 1)
        end_y = batch[-1].view(-1, 1).float()
        start_p = outputs[0]  # (batch_size, c_len)
        end_p = outputs[1]

        start_pred = []
        for _ in range(batch_size):
            start_pred.append(torch.arange(c_len).float())
        start_pred = torch.stack(start_pred).cuda()  # (batch_size, c_len)

        end_pred = []
        for _ in range(batch_size):
            end_pred.append(torch.arange(c_len).float())
        end_pred = torch.stack(end_pred).cuda()

        start = torch.max(start_y, start_pred)  # (batch_size, s_c_len)
        start = start.unsqueeze(2).expand(batch_size, c_len, c_len)  # (batch_size, s_c_len, e_c_len)
        end = torch.min(end_y, end_pred)
        end = end.unsqueeze(1).expand(batch_size, c_len, c_len)  # (batch_size, s_c_len, e_c_len)

        interval = end - start + 1
        mask = (interval <= 0)
        interval.masked_fill_(mask, 1)

        len_y = end_y - start_y + 1
        len_y = len_y.unsqueeze(2).expand(batch_size, c_len, c_len)

        start_pred = start_pred.unsqueeze(2).expand(batch_size, c_len, c_len)
        end_pred = end_pred.unsqueeze(1).expand(batch_size, c_len, c_len)
        len_pred = end_pred - start_pred + 1
        mask_1 = (len_pred <= 0)
        len_pred.masked_fill_(mask_1, 1)

        prec = interval / len_y
        rec = interval / len_pred
        score = 1 - ((1 + 1.2**2) * prec * rec) / (rec + 1.2**2 * prec)

        score.masked_fill_(mask, 1)
        score.masked_fill_(mask_1, 1)  # (batch_size, s_c_len, e_c_len)

        start_p = start_p.unsqueeze(2).expand(batch_size, c_len, c_len)
        end_p = end_p.unsqueeze(1).expand(batch_size, c_len, c_len)

        score = score * start_p * end_p
        score = torch.sum(score, dim=2)
        score = torch.sum(score, dim=1)
        score = torch.mean(score)

        loss_value = self.lam * loss_mle + (1-self.lam) * score

        return loss_value
