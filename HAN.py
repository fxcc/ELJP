# -*- coding: utf-8 -*-

""" 
Created at 2019-09-09 20:13:30
==============================
HAN 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttNet(nn.Module):
    def __init__(self, vocab_size, embed_size=256, dropout=0.1, hidden_size=256, embedding=None, padding_idx=None):
        super(WordAttNet, self).__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        if embedding is not None:
            self.emb.weight = nn.Parameter(embedding)
        else:
            nn.init.normal_(self.emb.weight, mean=0, std=embed_size ** -0.5)
            nn.init.constant_(self.emb.weight[padding_idx], 0)

        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)

        self.word_weight = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.context_weight = nn.Linear(2 * hidden_size, 1)
        self._init_linear_wt(self.word_weight)
        self._init_linear_wt(self.context_weight)

    def _init_linear_wt(self, linear):
        linear.weight.data.normal_(std=1e-4)
        if linear.bias is not None:
            linear.bias.data.normal_(std=1e-4)

    def forward(self, input, hidden_state):
        embedding = self.emb(input)
        f_output, h_output = self.gru(embedding, hidden_state)  # feature output and hidden state output
        output = self.word_weight(f_output)
        output = F.tanh(output)

        output = self.context_weight(output)
        output = F.tanh(output)

        output = F.softmax(output)
        output = f_output * output
        return output, h_output


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=256, word_hidden_size=256):
        super(SentAttNet, self).__init__()

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True, batch_first=True)

        self.sent_wight = nn.Linear(2 * sent_hidden_size, 2 * sent_hidden_size)
        self.context_weight = nn.Linear(2 * sent_hidden_size, 1)

        self._init_linear_wt(self.sent_wight)
        self._init_linear_wt(self.context_weight)

    def _init_linear_wt(self, linear):
        linear.weight.data.normal_(std=1e-4)
        if linear.bias is not None:
            linear.bias.data.normal_(std=1e-4)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)

        output = self.sent_wight(f_output)
        output = F.tanh(output)

        output = self.context_weight(output)
        output = F.tanh(output)

        output = F.softmax(output)
        output = f_output * output

        return output, h_output


class HierAttNet(nn.Module):
    def __init__(self, vocab_size, word_hidden_size, sent_hidden_size, num_classes, embed_size=256,
                 dropout=0.1, embedding=None, padding_idx=None):
        super(HierAttNet, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttNet(vocab_size, embed_size=embed_size, dropout=0.1, hidden_size=word_hidden_size,
                                       embedding=embedding, padding_idx=padding_idx)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)

    def forward(self, input):
        word_hidden_state = torch.zeros(2, input.size(0), self.word_hidden_size).to(input.device)
        sent_hidden_state = torch.zeros(2, input.size(0), self.sent_hidden_size).to(input.device)

        output_list = []
        for i in range(input.size(1)):
            output, word_hidden_state = self.word_att_net(input[:, i, :], word_hidden_state)
            output = torch.sum(output, 1).unsqueeze(1)
            output_list.append(output)
        output = torch.cat(output_list, 1)
        output, sent_hidden_state = self.sent_att_net(output, sent_hidden_state)
        output = torch.sum(output, 1)
        output = self.fc(output)
        return output


class HAN_with_sent_label(nn.Module):
    def __init__(self, vocab_size, word_hidden_size, sent_hidden_size, num_classes, embed_size=256, sentence_classes=2,
                 dropout=0.1, embedding=None, padding_idx=None):
        super(HAN_with_sent_label, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttNet(vocab_size, embed_size=embed_size, dropout=0.1, hidden_size=word_hidden_size,
                                       embedding=embedding, padding_idx=padding_idx)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
        self.fc = nn.Linear(2 * sent_hidden_size, sentence_classes)
        self.fc2 = nn.Linear(2 * sent_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, input):
        word_hidden_state = torch.zeros(2, input.size(0), self.word_hidden_size).to(input.device)
        sent_hidden_state = torch.zeros(2, input.size(0), self.sent_hidden_size).to(input.device)

        output_list = []
        for i in range(input.size(1)):
            output, word_hidden_state = self.word_att_net(input[:, i, :], word_hidden_state)
            output = torch.sum(output, 1).unsqueeze(1)
            output_list.append(output)
        output = torch.cat(output_list, 1)
        output, sent_hidden_state = self.sent_att_net(output, sent_hidden_state)

        doc_output = self.dropout(output)
        logits = self.fc(doc_output)

        logits = torch.sigmoid(logits)
        key_or_not = logits[:, :, 1]

        logits2 = torch.mul(key_or_not.unsqueeze(2), doc_output)

        sent_hidden_state = self.dropout(sent_hidden_state).permute(1, 0, 2).contiguous().view(input.size(0),-1).unsqueeze(1)

        output = torch.sum(torch.cat((sent_hidden_state, logits2), 1), 1)
        output = self.fc2(output)

        return output, logits
