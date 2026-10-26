# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Xuefeng Cui
import pickle
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors

from .definitions import *

def tokens_to_smiles(tokens):
    return [x.replace('L', 'Cl').replace('R', 'Br') for x in tokens]

def tokens_decode(tokens_mat, one_hot=True):
    if one_hot:
        tokens_mat = torch.argmax(tokens_mat, dim=2)

    tokens_ls = [[VOC[idx] for idx in sequence] for sequence in tokens_mat.tolist()]

    joined_tokens = [''.join(token) for token in tokens_ls]

    return joined_tokens

def tokens_crop_sequence(tokens_str, final_char=FINAL_CHAR):
    tokens_split = [s.split(final_char, 1)[0] for s in tokens_str]

    return tokens_split

def ctoy(c):
    return VOC2IDX.get(c, PAD_CHAR)

y_tokens = len(VOC)

def sequence_ytoc(seq):
    seq = tokens_decode(seq, one_hot=False)
    seq = tokens_crop_sequence(seq)
    seq = tokens_to_smiles(seq)
    return seq

def embed_ytox(y):
    return F.one_hot(y.to(torch.int64), num_classes=len(VOC)).float().unsqueeze(1)

def replace_smiles_ele(smiles):
    smiles = smiles.replace('Cl', 'L')
    smiles = smiles.replace('Br', 'R')
    return smiles

def recover_smiles_ele(smiles):
    smiles = smiles.replace('L', 'Cl')
    smiles = smiles.replace('R', 'Br')
    return smiles

def spilt_smiles(smiles):
    matches_ = re.findall(r'[^[]|\[.*?\]', smiles)
    return matches_

def pad_tokens_ls(tokens_ls):
    pad_ls = [INITIAL_CHAR]
    pad_ls.extend(tokens_ls)
    pad_ls.append(FINAL_CHAR)
    pad_ls.extend([PAD_CHAR] * (SEQUENCE_LEN - len(pad_ls)))
    return pad_ls[:SEQUENCE_LEN]

def smiles2tokens(smiles):
    smiles = replace_smiles_ele(smiles)
    tokens_ls = spilt_smiles(smiles)

    pad_ls = pad_tokens_ls(tokens_ls)
    return pad_ls



def tokens2idx(tokens_ls):
    return [VOC2IDX.get(c, 0) for c in tokens_ls]

def smiles2idx(smiles):
    tokens_ls = smiles2tokens(smiles)
    idx_ls = tokens2idx(tokens_ls)
    return idx_ls

def idx2vec(idx_ls):
    num_classes = len(VOC)
    return np.eye(num_classes)[np.array(idx_ls)].astype(np.float32)

def tokens2vec(tokens_ls):
    idx_ls = tokens2idx(tokens_ls)
    return idx2vec(idx_ls)


def smiles2_gt_lab(smiles):
    idx_ls = smiles2idx(smiles)  # (128,)
    gt_ls = idx_ls[:-1]
    lab_ls = idx_ls[1:]
    gt = idx2vec(gt_ls)  # (127, 39)
    return gt, np.array(lab_ls, dtype=np.float32)  # (seq_len, voc_size), (seq_len,)


def remove_pad(sm):
    if sm[0] == INITIAL_CHAR:
        sm = sm[1:]
    end_index = sm.find(FINAL_CHAR)
    if end_index != -1:
        sm = sm[:end_index]
    return sm



def idx2smiles(sm_idx):
    sm_pad = ''.join(VOC[idx] for idx in sm_idx.tolist())
    smiles = remove_pad(sm_pad)

    replace_vec = np.vectorize(recover_smiles_ele)
    smiles = str(replace_vec(smiles))  # 把smiles中的L，R替换为Cl，Br
    return smiles
