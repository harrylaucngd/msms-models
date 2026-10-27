"""Random molecular graphs for each query's formula (MassSpecGym ``RandomDeNovo``).

For the formula-known challenge the formula comes from the query, which equals the
``CalcMolFormula`` of the target that MassSpecGym used. Python's ``random`` and NumPy are
seeded as ``pytorch_lightning.seed_everything`` did, and queries are generated in order, so a
formula seen before returns its cached molecules.
"""

from __future__ import annotations


def predict(config, queries):
    import pickle
    import random

    import numpy as np
    from rdkit import Chem

    from .vendor.random_generator import RandomDeNovo

    p = config.parameters
    generator = RandomDeNovo(
        formula_known=True,
        count_of_valid_valence_assignments=int(p.get("count_of_valid_valence_assignments", 10)),
        estimate_chem_element_stats=p.get("estimate_chem_element_stats", False),
        max_top_k=int(p.get("max_top_k", 10)),
        enforce_connectivity=p.get("enforce_connectivity", True),
        cache_results=p.get("cache_results", True),
    )
    if generator.estimate_chem_element_stats:
        with open(config.require_assets("element_stats")["element_stats"], "rb") as handle:
            generator.element_2_bond_stats = pickle.load(handle)["element_2_bond_stats"]
    random.seed(config.seed)
    np.random.seed(config.seed)
    rows = []
    for query in queries:
        molecules = generator.generate_random_molecule_graphs_via_traversal(query["formula"])
        for molecule in molecules:
            Chem.RemoveHs(molecule)  # as upstream, the result is discarded: hydrogens stay explicit
        rows.append({"id": query["id"], "candidates": [Chem.MolToSmiles(m) for m in molecules],
                     "status": "ok"})
    return rows
