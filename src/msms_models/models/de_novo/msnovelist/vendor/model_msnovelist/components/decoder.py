# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Xuefeng Cui
import torch
from torch import nn
from torch.nn import init


class SequenceDecoder(nn.Module):
    def __init__(self, input_size=306, hidden_size=256, output_size=39, num_layers=3, return_states=False):
        super(SequenceDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_states = return_states

        self.batch_norm = nn.BatchNorm1d(input_size, eps=1e-3, momentum=0.01)
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, self.output_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)  # Xavier Uniform initialization
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # Orthogonal initialization is a common choice
            elif 'bias' in name:
                param.data.fill_(0)  # Bias initialized to zero

        init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            init.constant_(self.dense.bias, 0.0)

    def forward(self, inputs, initial_state=None):
        """
        :param inputs: (batch_size, seq_len=127, hid_size=306=256+11+39)
        :param initial_state:
                (h, c) h,c: (num_layers=3, batch_size, hid_size=256)
        :return:
            out: (batch_size, seq_len=127, out_size=39)
            states: (h, c) h,c: (num_layers=3, batch_size, hid_size=256)
        """
        x = torch.cat(inputs, dim=-1).float()
        batch_size, seq_len, voc_size = x.size()
        x = self.batch_norm(x.view(batch_size * seq_len, -1)).view(x.size())
        x, states = self.lstm(x, initial_state)

        out = self.dense(x)

        if self.return_states:
            return out, states
        else:
            return out
