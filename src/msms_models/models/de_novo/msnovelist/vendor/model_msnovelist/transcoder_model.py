# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Xuefeng Cui
import numpy as np
import torch
from torch import nn

from .blueprinted_model import BlueprintedModel

class TranscoderModel(BlueprintedModel):
    def __init__(self, **kwargs):
        super(TranscoderModel, self).__init__(**kwargs)

    def forward(self, inputs):
        formula = inputs['MF']  # (batch_size=256, 10)
        fingerprints = inputs['FP']  # (batch_size=256, 3609)
        fingerprints_ = self.fingerprint_rounding(fingerprints)

        z, rnn_states = self.encoder([fingerprints_, formula])

        rnn_states = [item.detach() for item in rnn_states]
        estimated_h_count = self.hydrogen_estimator(inputs['tokens_X'])

        estimated_h_sum = self.hydrogen_sum(estimated_h_count)

        estimated_h_count_ = self.gradient_stop(estimated_h_count)
        auxiliary_counter_input = torch.cat([inputs['tokens_X'], estimated_h_count_], dim=-1)

        auxiliary_counter_input_transformed = self.auxiliary_counter_input_transformer(
            auxiliary_counter_input)

        auxiliary_counter_start_states = self.auxiliary_counter_start_state_transformer(inputs['MF'])

        element_grammar_count = self.auxiliary_counter(
            auxiliary_counter_input_transformed,
            initial_state=auxiliary_counter_start_states)

        z_repeated = self.z_time_transformer(z)

        decoder_input = [inputs['tokens_X'],  # (batch_size, seq_len, 39)
                         element_grammar_count,  # (batch_size, seq_len, 11)
                         z_repeated]    # (batch_size, seq_len, 256)

        decoder_out = self.sequence_decoder(decoder_input, rnn_states)

        return decoder_out, estimated_h_sum  # , element_grammar_count
