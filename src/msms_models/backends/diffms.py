"""Formula-conditioned DiffMS with the encoder from the full trained checkpoint."""

from __future__ import annotations

import json
import re
from collections import Counter
from types import SimpleNamespace

from ..io import atomic_json
from ..posteriors import get_posteriors


def load_model(config):
    import torch
    from mist.models.scaled_mist import ScaledMIST
    from torch import nn

    from .._vendor.diffms.diffusion.extra_features import ExtraFeatures
    from .._vendor.diffms.diffusion.extra_features_molecular import ExtraMolecularFeatures
    from .._vendor.diffms.diffusion.noise_schedule import (
        MarginalUniformTransition,
        PredefinedNoiseScheduleDiscrete,
    )
    from .._vendor.diffms.models.transformer_model import GraphTransformer
    from .._vendor.diffms.molecules import MolecularVisualization
    from .._vendor.diffms.sampling import NativeSampling
    from .._vendor.diffms.utils import PlaceHolder

    assets = config.require_assets("checkpoint", "model_config")
    metadata = json.loads(assets["model_config"].read_text())
    if metadata.get("dataset") != config.dataset or metadata.get("schema_version") != 1:
        raise ValueError("DiffMS model_config has the wrong dataset or schema")
    if metadata.get("encoder_type") != "scaled-mist":
        raise ValueError("This DiffMS bundle requires its scaled-MIST encoder architecture")
    architecture = metadata["decoder"]
    info = SimpleNamespace(**metadata["dataset_info"])
    info.atom_weights = {int(k): v for k, v in info.atom_weights.items()}

    class Model(NativeSampling, nn.Module):
        @property
        def device(self):
            return next(self.parameters()).device

    model = Model()
    model.decoder = GraphTransformer(**architecture, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU())
    model.encoder = ScaledMIST(**metadata["encoder"])
    model.merge_function = nn.Linear(4096, metadata["conditioning_bits"])
    model.T = int(metadata["diffusion_steps"])
    model.noise_schedule = PredefinedNoiseScheduleDiscrete(
        metadata["noise_schedule"], timesteps=model.T
    )
    node = torch.tensor(info.node_types, dtype=torch.float32)
    node = node / node.sum()
    edge = torch.tensor(info.edge_types, dtype=torch.float32)
    edge = edge / edge.sum()
    model.Xdim_output = architecture["output_dims"]["X"]
    model.Edim_output = architecture["output_dims"]["E"]
    model.ydim_output = architecture["output_dims"]["y"]
    model.transition_model = MarginalUniformTransition(node, edge, model.ydim_output)
    model.limit_dist = PlaceHolder(
        X=node, E=edge, y=torch.ones(model.ydim_output) / max(model.ydim_output, 1)
    )
    model.extra_features = ExtraFeatures(metadata["extra_features"], info)
    model.domain_features = ExtraMolecularFeatures(info)
    model.visualization_tools = MolecularVisualization(info.remove_h, info)
    model.dataset_info = info
    model.inference_bf16 = False
    payload = torch.load(assets["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    # No separately pretrained encoder or decoder is loaded over these weights.
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    return model.to(config.device).eval(), metadata


def formula_graph(formula, atom_decoder, conditioning):
    import torch
    from torch_geometric.data import Data

    pieces = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if "".join(e + n for e, n in pieces) != formula:
        raise ValueError("Formula is not a neutral elemental sum")
    counts = Counter()
    for element, count in pieces:
        counts[element] += int(count or 1)
    unsupported = set(counts) - set(atom_decoder)
    if unsupported:
        raise ValueError(f"Unsupported DiffMS elements: {sorted(unsupported)}")
    # Native GraphFeaturizer parses SMILES without Chem.AddHs: formula
    # hydrogens are implicit, rather than six extra graph nodes for ethanol.
    atom_ids = [
        i
        for i, element in enumerate(atom_decoder)
        if element != "H"
        for _ in range(counts[element])
    ]
    if not atom_ids:
        raise ValueError("Empty formula")
    x = torch.nn.functional.one_hot(torch.tensor(atom_ids), len(atom_decoder)).float()
    return Data(
        x=x,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 5)),
        y=conditioning.detach().cpu().reshape(1, -1),
    )


def rank_molecules(molecules):
    from rdkit import Chem

    from .._vendor.diffms.utils import is_valid

    inchis = []
    for molecule in molecules:
        if molecule is None or not is_valid(molecule):
            continue
        try:
            value = Chem.MolToInchi(molecule)
            if value:
                inchis.append(value)
        except (ValueError, RuntimeError):
            continue
    ranked = Counter(inchis).most_common()
    return [Chem.MolToSmiles(Chem.MolFromInchi(i)) for i, _ in ranked], [
        count for _, count in ranked
    ]


def predict(config, queries):
    import torch
    from torch_geometric.data import Batch

    torch.set_num_threads(config.workers)
    model, metadata = load_model(config)
    posterior = get_posteriors(config, queries, encoder=model.encoder)
    samples, chunk = (
        int(config.parameters.get("num_samples", 100)),
        int(config.parameters.get("sample_batch_size", 10)),
    )
    if min(samples, chunk) < 1:
        raise ValueError("DiffMS sample counts must be positive")
    rows = []
    with torch.inference_mode():
        conditioning = model.merge_function(torch.tensor(posterior, device=config.device))
        for index, (query, vector) in enumerate(zip(queries, conditioning, strict=True)):
            try:
                graph = formula_graph(
                    query["formula"], metadata["dataset_info"]["atom_decoder"], vector
                )
            except ValueError as error:
                rows.append(
                    {
                        "id": query["id"],
                        "candidates": [],
                        "status": "unsupported_formula",
                        "reason": str(error),
                    }
                )
                continue
            torch.manual_seed(config.seed + index)
            generated = []
            for start in range(0, samples, chunk):
                batch = Batch.from_data_list([graph] * min(chunk, samples - start)).to(
                    config.device
                )
                generated.extend(model.sample_batch(batch))
            candidates, counts = rank_molecules(generated)
            rows.append(
                {"id": query["id"], "candidates": candidates, "scores": counts, "status": "ok"}
            )
    atomic_json(
        config.output / "decoder_protocol.json",
        {
            "decoder": "DiffMS",
            "num_samples": samples,
            "sample_batch_size": chunk,
            "diffusion_steps": model.T,
            "atom_order": metadata["dataset_info"]["atom_decoder"],
            "ranking": "native InChI sample frequency, first occurrence breaks ties",
            "checkpoint_loading": "strict full encoder, projection and decoder state",
            "input_bonds": "none; all bonds are generated from diffusion noise",
            "hydrogens": "implicit; graph nodes follow heavy-atom formula counts",
        },
    )
    return rows
