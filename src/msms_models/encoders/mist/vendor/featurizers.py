"""featurizers.py

msms-models: the evaluation path of DiffMS ``PeakFormula`` (``src/mist/data/featurizers.py``,
coleygroup/DiffMS @9d9f4fd). The subformula tree is passed in instead of being looked up
from a folder glob; training-time augmentation, MAGMA supervision, peak capping and the
unused fragment m/z vector are omitted. The remaining statements are verbatim.
"""

from typing import Dict, List

import numpy as np
import torch

from . import chem_utils as utils

cat_types = {"frags": 0, "loss": 1, "ab_loss": 2, "cls": 3}


def get_peak_dict(tree: dict) -> dict:
    """Body of ``PeakFormula._get_peak_dict`` after ``json.load`` (``max_peaks=None``)."""
    root_form = tree["cand_form"]
    root_ion = tree["cand_ion"]
    output_tbl = tree["output_tbl"]

    if output_tbl is None:
        frags = []
        intens = []
        ions = []
    else:
        frags = output_tbl["formula"]
        intens = output_tbl["ms2_inten"]
        ions = output_tbl["ions"]

    out_dict = {
        "frags": frags,
        "intens": intens,
        "ions": ions,
        "root_form": root_form,
        "root_ion": root_ion,
    }
    return out_dict


def featurize(
    tree: dict,
    spec_name: str,
    instrument: str = "",
    cls_type: str = "ms1",
    inten_transform: str = "float",
) -> Dict:
    """``PeakFormula._featurize`` with ``train_mode=False``."""
    peak_dict = get_peak_dict(tree)

    # Add in chemical formuale
    root = peak_dict["root_form"]

    forms_vec = [utils.formula_to_dense(i) for i in peak_dict["frags"]]
    root_vec = utils.formula_to_dense(root)
    root_ion = utils.get_ion_idx(peak_dict["root_ion"])
    inten_vec = list(peak_dict["intens"])
    ion_vec = [utils.get_ion_idx(i) for i in peak_dict["ions"]]
    type_vec = len(forms_vec) * [cat_types["frags"]]
    instrument = utils.get_instr_idx(instrument)

    if cls_type == "ms1":
        cls_ind = cat_types.get("cls")
        inten_vec.append(1.0)
        type_vec.append(cls_ind)
        forms_vec.append(root_vec)
        ion_vec.append(root_ion)

    elif cls_type == "zeros":
        cls_ind = cat_types.get("cls")
        inten_vec.append(0.0)
        type_vec.append(cls_ind)
        forms_vec.append(np.zeros_like(root_vec))
        ion_vec.append(root_ion)
    else:
        raise NotImplementedError()

    # Featurize all formulae
    inten_vec = np.array(inten_vec)
    if inten_transform == "float":
        pass
    else:
        # msms-models: only the "float" transform of the released checkpoints is kept.
        raise NotImplementedError()

    forms_vec = np.array(forms_vec)  # / utils.NORM_VEC[None, :]

    out_dict = {
        "peak_type": np.array(type_vec),
        "form_vec": forms_vec,
        "ion_vec": ion_vec,
        "frag_intens": inten_vec,
        "name": spec_name,
        "instrument": instrument,
    }
    return out_dict


def collate_fn(input_list: List[dict]) -> Dict:
    """``PeakFormula.collate_fn`` without MAGMA fingerprints: right-pads with zeros."""
    # Determines the number of channels
    names = [j["name"] for j in input_list]
    peak_form_tensors = [torch.from_numpy(j["form_vec"]) for j in input_list]
    inten_tensors = [torch.from_numpy(j["frag_intens"]) for j in input_list]
    type_tensors = [torch.from_numpy(j["peak_type"]) for j in input_list]
    instrument_tensors = torch.FloatTensor([j["instrument"] for j in input_list])
    ion_tensors = [torch.FloatTensor(j["ion_vec"]) for j in input_list]

    peak_form_lens = np.array([i.shape[0] for i in peak_form_tensors])
    max_len = np.max(peak_form_lens)
    padding_amts = max_len - peak_form_lens

    type_tensors = [
        torch.nn.functional.pad(i, (0, pad_len))
        for i, pad_len in zip(type_tensors, padding_amts)
    ]
    ion_tensors = [
        torch.nn.functional.pad(i, (0, pad_len))
        for i, pad_len in zip(ion_tensors, padding_amts)
    ]
    inten_tensors = [
        torch.nn.functional.pad(i, (0, pad_len))
        for i, pad_len in zip(inten_tensors, padding_amts)
    ]
    peak_form_tensors = [
        torch.nn.functional.pad(i, (0, 0, 0, pad_len))
        for i, pad_len in zip(peak_form_tensors, padding_amts)
    ]

    # Stack everything (bxd for root, bxp for others)
    type_tensors = torch.stack(type_tensors, dim=0).long()
    peak_form_tensors = torch.stack(peak_form_tensors, dim=0).float()
    ion_tensors = torch.stack(ion_tensors, dim=0).float()

    inten_tensors = torch.stack(inten_tensors, dim=0).float()
    num_peaks = torch.from_numpy(peak_form_lens).long()

    return_dict = {
        "types": type_tensors,
        "form_vec": peak_form_tensors,
        "ion_vec": ion_tensors,
        "intens": inten_tensors,
        "names": names,
        "num_peaks": num_peaks,
        "instruments": instrument_tensors,
    }
    return return_dict
