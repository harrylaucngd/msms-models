# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Xuefeng Cui
import torch
from torch import nn
from torch.nn import init


class HydrogenEstimator(nn.Module):
    def __init__(self, layers=2, units=32, pad_term_mask=None, return_states=False):
        super(HydrogenEstimator, self).__init__()
        self.layers_ = layers
        self.units_ = units
        self.register_buffer("pad_term_mask", pad_term_mask, persistent=False)
        self.return_states = return_states

        self.lstm = nn.LSTM(input_size=39, hidden_size=self.units_, num_layers=self.layers_, batch_first=True)
        self.out_layer = nn.Linear(self.units_, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)  # Xavier Uniform initialization
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # Orthogonal initialization is a common choice
            elif 'bias' in name:
                param.data.fill_(0)  # Bias initialized to zero

        init.xavier_uniform_(self.out_layer.weight)
        if self.out_layer.bias is not None:
            init.constant_(self.out_layer.bias, 0.0)

    def forward(self, inputs, initial_state=None):
        """
        :param inputs: (batch_size, seq_len, voc_size)  (256,127,39)
        :param initial_state:
                1-test、(h, c) h,c: (num_layers=2, batch_size, hid_size=32)
                2-train、None
        :return:
            layer_stack: (batch_size, seq_len=127, out_size=1)
            states: (h, c) h,c: (num_layers=2, batch_size, hid_size=32)
        """
        tokens_input = inputs

        layer_stack, states = self.lstm(tokens_input, initial_state)

        layer_stack = self.out_layer(layer_stack)
        if self.pad_term_mask is not None:
            mask_vec = torch.max(tokens_input * self.pad_term_mask, dim=2, keepdim=True)[0]
            layer_stack = layer_stack * mask_vec

        if self.return_states:
            return layer_stack, states
        else:
            return layer_stack  # (batch_size, seq_len, out_size)
