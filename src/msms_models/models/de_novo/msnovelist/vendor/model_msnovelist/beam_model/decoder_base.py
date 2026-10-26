# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Xuefeng Cui
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from ... import tokens_process

class DecoderBase:
    def __init__(self, encoder, decoder, steps, n, k, kk, temperature=1., config=None):
        if config is None:
            config = {}
        self.config = {
            'tokenization': 'smiles_process',
            'clip_invalid_counts': True,
            'sequence_length_correction': False
        }
        if "decoder_config" in config:
            config_ = config["decoder_config"]
            self.config.update(config_)

        self.tokenization = tokens_process
        self.clip_invalid_counts = self.config['clip_invalid_counts']

        self.final_char = self.tokenization.ctoy(self.tokenization.FINAL_CHAR)
        self.initial_char = self.tokenization.ctoy(self.tokenization.INITIAL_CHAR)
        self.pad_char = self.tokenization.ctoy(self.tokenization.PAD_CHAR)

        self.encoder = encoder
        self.steps = steps
        self.n = n
        self.decoder = decoder
        self.k = k
        self.kk = kk
        self.y_tokens = torch.tensor(self.tokenization.y_tokens)
        self.y_init, self.scores_init, self.pad_mask = self._init_templates()
        self.eps = 0.001
        self.temperature = temperature
        self.clip_invalid_counts_factor = 1. * self.clip_invalid_counts
        self.sequence_length_correction = self.config['sequence_length_correction']

    def softmax(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

    def _init_templates(self):
        raise NotImplementedError("Not available in abstract base class")

    def beam_step(self, y, states, scores, counts):
        raise NotImplementedError("Not available in abstract base class")

    def decode_beam(self, states_init):
        xstep = self.tokenization.embed_ytox(self.y_init)
        scores = self.scores_init  # (1024=8*128,)

        states = states_init['states']  # (1024=8*128, 1931)
        y_chain = []
        sequences_chain = []
        scores_chain = []
        i = 0

        while i < self.steps:
            decoder_out = self.decoder({'tokens_X': xstep.to(next(self.decoder.parameters()).device), 'states': states.to(next(self.decoder.parameters()).device)})
            decoder_out = {k: v.cpu() for k, v in decoder_out.items()}
            y = decoder_out['tokens_X']  # (1024=8*128,1,39)
            states = decoder_out['states']  # (1024=8*128,1931)
            counts = decoder_out['counts']  # (1024=8*128,11)

            y = self.softmax(y)
            ymax, ysequence, states, scores = self.beam_step(y, states, scores, counts)
            xstep = self.tokenization.embed_ytox(ymax)

            y_chain.append(ymax)
            sequences_chain.append(ysequence)
            scores_chain.append(scores)
            i += 1

        sequences_final = torch.stack(sequences_chain)
        y_final = torch.stack(y_chain)
        scores_final = torch.stack(scores_chain)

        return sequences_final, y_final, scores_final

    def beam_traceback(self, sequences, y, scores, reverse=True):
        r_t_r = lambda x: x.reshape(-1, self.n, self.k).permute(1, 2, 0).reshape(self.n, -1)
        y_ = r_t_r(y)  # (8,128*127)
        sc_ = r_t_r(scores)  # (8,128*127)

        ends = torch.where(y_ == self.final_char, sc_, torch.full_like(sc_, -np.inf))
        top_kk = torch.topk(ends, self.kk, dim=1)  # (8,128)
        scores = top_kk.values  # (8,128) 8个小分子的128候选序列的得分
        top_kk_vec_ = np.unravel_index(top_kk.indices.view(-1), (self.k, self.steps))
        top_kk_vec = tuple([torch.from_numpy(i) for i in top_kk_vec_])

        i_source = torch.arange(self.n * self.kk) // self.kk  # (1024,)  这里的代码默认 k==kk
        pos = top_kk_vec[0] + self.k * i_source  # (1024,)  横坐标
        step = top_kk_vec[1]
        length = step

        max_length = step.max() + 1
        traces = torch.zeros((max_length, self.n * self.kk), dtype=torch.int32)

        i = 0
        while (step > 0).any():
            token_at_pos = y[step, pos]
            continue_at = sequences[step, pos]
            pos = continue_at
            traces[i] = token_at_pos
            i += 1
            step = torch.clamp(step - 1, min=0)

        traces = traces.t()
        traces = self.sequence_mask(length + 1, max_length, dtype=torch.bool) * traces

        length_with_termination_capped = torch.min(max_length, length + 1)

        if not reverse:
            return traces, scores, length + 1

        traces = self.reverse_sequence(traces, length_with_termination_capped)

        return traces, scores, length + 1

    def sequence_mask(self, lengths, max_len, dtype=torch.bool):
        """
        功能类似于TensorFlow中的tf.sequence_mask。

        Args:
        lengths: 序列的实际长度，形状为 (batch_size,) 的张量。
        maxlen: 掩码矩阵的最大长度，如果为None，则使用lengths中的最大值。
        dtype: 掩码的类型，默认是torch.bool。

        Returns:
        一个形状为 (batch_size, maxlen) 的布尔张量。
        """
        row_vector = torch.arange(0, max_len, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        return mask.to(dtype)

    def reverse_sequence(self, traces, length_with_termination_capped):
        """
        :param traces: tensor (num_seq, max_length)
            traces = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        :param length_with_termination_capped: tensor (num_seq,)
            length_with_termination_capped = torch.tensor([3, 4])
        :return:
        """
        reversed_traces = torch.zeros_like(traces)
        for i, length in enumerate(length_with_termination_capped):
            reversed_traces[i, :length] = torch.flip(traces[i, :length], [0])

        for i, length in enumerate(length_with_termination_capped):
            reversed_traces[i, length:] = traces[i, length:]
        return reversed_traces

    def sequence_ytoc(self, seq):
        return self.tokenization.sequence_ytoc(seq)
