"""Native formula-constrained FOAM optimization from explicit offline seeds."""

from __future__ import annotations

import random

import numpy as np
from rdkit import Chem

from ....data.candidates import load_candidates
from ....simulators.iceberg import load_model


class SpectrumOracle:
    """FOAM's entropy/SA objectives, without a known target structure."""

    multiobj = True
    smi = None
    mol = None
    self_iceberg_scores = None
    sa_score_target = None
    criteria = "entropy"

    def __init__(self, model, config, query, seeds):
        from ....simulators.iceberg.vendor import common

        self.model, self.config, self.query = model, config, query
        self.formula = query["formula"]
        self.num_bins = int(config.parameters.get("num_bins", 15000))
        self.upper_limit = float(config.parameters.get("upper_limit", 1500))
        raw = np.column_stack([query["spectrum"]["mz"], query["spectrum"]["intensity"]])
        # FOAM consumes preprocessed native intensities; preprocessing is explicit.
        self.reference = common.bin_spectra([raw], self.num_bins, self.upper_limit)[0]
        self.ce = query["collision_energy"]
        self.seeds = list(seeds)
        self.cache = {}

    @staticmethod
    def oracle_name():
        return "ICEBERGWithSAOracle"

    def conditional_info(self):
        return {"formula": self.formula, "morgan_fp": None}

    def _get_fp(self):
        return None

    def score_valid_mols(self, examples):
        import torch
        from rdkit.Contrib.SA_Score import sascorer

        from .vendor.distance import dist_bin
        from .vendor.utils import tautomerize_smi

        strings = [s if isinstance(s, str) else Chem.MolToSmiles(s) for s in examples]
        output = []
        for string in strings:
            normalized = tautomerize_smi(string)
            if normalized not in self.cache:
                with torch.inference_mode():
                    prediction = self.model.predict_mol(
                        normalized,
                        self.ce,
                        self.query["precursor_mz"],
                        self.query["adduct"],
                        0.0,
                        self.config.device,
                        int(self.config.parameters.get("max_nodes", 100)),
                        instrument=self.query["instrument"],
                        adduct_shift=True,
                        binned_out=True,
                    )["spec"]
                if isinstance(prediction, torch.Tensor):
                    prediction = prediction.detach().cpu().numpy()
                prediction = np.asarray(prediction)
                ignore = (
                    (self.query["precursor_mz"] - 1) * (self.num_bins / self.upper_limit)
                    if self.config.parameters.get("ignore_precursor_peak", True)
                    else None
                )
                distance = dist_bin(
                    [{self.ce: prediction}],
                    {self.ce: self.reference},
                    sparse=False,
                    ignore_peak=ignore,
                    func="entropy",
                )[0]
                distance = float(np.round(distance, 6))
                sa = 1 - sascorer.calculateScore(Chem.MolFromSmiles(normalized)) / 10
                self.cache[normalized] = (1 - distance, sa)
            output.append(self.cache[normalized])
        return np.array(output, dtype=float).reshape(-1, 2).T

    def __call__(self, molecules):
        output = np.zeros((2, len(molecules)))
        positions = [i for i, m in enumerate(molecules) if m is not None]
        if positions:
            output[:, positions] = self.score_valid_mols([molecules[i] for i in positions])
        return output

    def get_seed_smiles(self, max_possible=200, top_k=20):
        scores = self.score_valid_mols(self.seeds)
        if max_possible < 0 or max_possible >= len(self.seeds):
            selected = np.arange(len(self.seeds))
        else:
            # Native seed selection keeps the strongest entropy candidates and
            # samples the remainder. Negative size explicitly means all seeds.
            order = np.argsort(scores[0])
            top_k = min(top_k, max_possible)
            random_count = max_possible - top_k
            others = (
                np.random.choice(order[:-top_k], random_count, replace=False)
                if random_count
                else []
            )
            selected = np.concatenate([order[-top_k:], others]).astype(int)
        return (
            [self.seeds[i] for i in selected],
            scores[0, selected].tolist(),
            scores[1, selected].tolist(),
        )


def predict(config, queries):
    import torch

    from .vendor.opt_graph_ga_fc.graph_ga_fc import GraphGAFCOptimizer
    from .vendor.utils import tautomerize_smi

    paths = config.require_assets("seeds")
    seeds = load_candidates(paths["seeds"], queries)
    torch.set_num_threads(config.workers)
    model = load_model(config)
    p = config.parameters
    rows = []
    for index, query in enumerate(queries):
        random.seed(config.seed + index)
        np.random.seed(config.seed + index)
        torch.manual_seed(config.seed + index)
        strings = []
        for candidate in seeds[query["id"]]:
            mol = Chem.MolFromSmiles(tautomerize_smi(candidate))
            Chem.RemoveStereochemistry(mol)
            strings.append(Chem.MolToSmiles(mol))
        directory = config.output / "optimization" / f"{index:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        oracle = SpectrumOracle(model, config, query, strings)
        optimizer = GraphGAFCOptimizer(
            oracle=oracle,
            save_dir=str(directory),
            num_workers=config.workers,
            max_calls=int(p.get("max_calls", 1500)),
            keep_population=int(p.get("keep_population", 20000)),
            population_size=int(p.get("population_size", 200)),
            offspring_size=int(p.get("offspring_size", 600)),
            starting_seed_size=int(p.get("starting_seed_size", -1)),
            num_islands=int(p.get("num_islands", 1)),
            mutation_rate=float(p.get("mutation_rate", 0.2)),
            criteria="entropy",
            truncate=p.get("truncate", True),
            selection_sorting_type="cand_crowding",
            parent_tiebreak="cand_crowding",
            wandb_mode="disable",
            eval_names=[],
            log_freq=int(p.get("log_frequency", 100)),
            patience=int(p.get("patience", 10)),
        )
        result = optimizer.optimize()
        # Native final reporting sorts all visited candidates by entropy alone.
        ranked = sorted(result["mol_buffer"].items(), key=lambda x: x[1]["scores"][0], reverse=True)
        rows.append(
            {
                "id": query["id"],
                "candidates": [s for s, _ in ranked],
                "scores": [float(v["scores"][0]) for _, v in ranked],
                "status": "ok",
                "objective_calls": optimizer.calls_made,
                "initial_seed_count": len(strings),
            }
        )
    return rows
