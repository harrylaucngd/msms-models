# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Xuefeng Cui
import torch
import numpy as np
import re
import itertools

INITIAL_CHAR = "$"
FINAL_CHAR = "*"
PAD_CHAR = "&"
SEQUENCE_LEN = 128

VOC = ['O',
       '=',
       'C',
       '1',
       'c',
       '2',
       '-',
       '3',
       '4',
       '(',
       ')',
       'n',
       '5',
       '6',
       '[nH]',
       'o',
       'N',
       '[N+]',
       '[O-]',
       'L',
       '[NH+]',
       'S',
       'F',
       's',
       'R',
       '#',
       'P',
       'I',
       '[N-]',
       '7',
       'p',
       '[n+]',
       '[NH3+]',
       '[C-]',
       '[NH2+]',
       '[H]']

VOC.extend([INITIAL_CHAR, FINAL_CHAR, PAD_CHAR])
VOC2IDX = {s: i for i, s in enumerate(VOC)}
IDX2VOC = {i: s for i, s in enumerate(VOC)}

VOC_UPPER = [x.upper() for x in VOC]

ELEMENTS = ['C', 'F', 'I', 'L', 'N', 'O', 'P', 'R', 'S']
ELEMENTS_RDKIT = ['C', 'F', 'I', 'Cl', 'N', 'O', 'P', 'Br', 'S', 'H']

elements_vec = lambda x: [e in x for e in ELEMENTS]
ELEMENT_MAP = torch.tensor(
    np.array([elements_vec(x) for x in VOC_UPPER], dtype=np.int32),
    dtype=torch.float32
)

GRAMMAR = [{'(': -1, ')': 1}]
GRAMMAR_MAP = torch.tensor(
    np.stack([np.sum(np.array([[val * (key in x) for key, val in grammar_.items()] for x in VOC]), axis=1)
              for grammar_ in GRAMMAR], axis=1),
    dtype=torch.float32)
