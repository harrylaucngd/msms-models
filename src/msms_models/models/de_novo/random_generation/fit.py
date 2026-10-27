"""Record the element and bond statistics of the training molecules for random generation.

The equivalent of MassSpecGym's one training epoch of ``RandomDeNovo`` with
``estimate_chem_element_stats=True``: ``training_step`` over every training spectrum's
molecule, then ``on_train_end``. The statistics are pickled for prediction:

    python -m msms_models.models.de_novo.random_generation.fit \\
        --train data/msg/train.jsonl --output checkpoints/random_generation/element_stats.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from ....data.io import load_queries


def fit(queries):
    from .vendor.random_generator import RandomDeNovo

    generator = RandomDeNovo(estimate_chem_element_stats=True)
    # Counts are per spectrum, as in a training epoch over the spectra.
    generator.training_step({"mol": [q["target_smiles"] for q in queries]})
    generator.on_train_end()
    return {
        "element_2_bond_stats": generator.element_2_bond_stats,
        "mol_weight_2_formulas": generator.mol_weight_2_formulas,
        "mol_weight_trn_values": generator.mol_weight_trn_values,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="Training queries with target_smiles")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    queries = load_queries(args.train)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(fit(queries), handle)
    print(json.dumps({"output": str(output), "molecules": len(queries)}))


if __name__ == "__main__":
    main()
