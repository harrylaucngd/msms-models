"""Registered models. Reading this table imports no model runtime.

Each ``module`` implements ``predict(config, queries)``, returning one ordered
prediction record per query. A model's assets and parameters are documented in its
guide and set in its configuration; variants of a model are configurations, not code.
"""

from dataclasses import asdict, dataclass
from importlib import import_module


@dataclass(frozen=True)
class Model:
    name: str
    task: str  # "retrieval" or "de-novo"
    module: str

    def to_dict(self):
        return asdict(self)

    def implementation(self):
        return import_module(self.module)


_PACKAGE = "msms_models.models"
MODELS: dict[str, Model] = {m.name: m for m in (
    Model("nearest-neighbor", "retrieval", f"{_PACKAGE}.retrieval.nearest_neighbor.inference"),
    Model("mist", "retrieval", f"{_PACKAGE}.retrieval.mist"),
    Model("iceberg", "retrieval", f"{_PACKAGE}.retrieval.iceberg"),
    Model("glacier", "retrieval", f"{_PACKAGE}.retrieval.glacier"),
    Model("dreams-chemberta", "retrieval", f"{_PACKAGE}.retrieval.dreams_chemberta.inference"),
    Model("chemformer-two-stage", "retrieval",
          f"{_PACKAGE}.retrieval.chemformer_two_stage.inference"),
    # Spectrum-blind stress tests (paper Appendix A.4)
    Model("chemberta-classifier", "retrieval",
          f"{_PACKAGE}.retrieval.chemberta_classifier.inference"),
    Model("pubchem-order", "retrieval", f"{_PACKAGE}.retrieval.pubchem_order.inference"),
    Model("chiral-atoms", "retrieval", f"{_PACKAGE}.retrieval.chiral_atoms"),
    # De novo generation
    Model("msnovelist", "de-novo", f"{_PACKAGE}.de_novo.msnovelist.inference"),
    Model("molforge", "de-novo", f"{_PACKAGE}.de_novo.molforge.inference"),
    Model("ms-gpt-base", "de-novo", f"{_PACKAGE}.de_novo.ms_gpt.inference"),
    Model("frigid-base", "de-novo", f"{_PACKAGE}.de_novo.frigid.inference"),
    Model("frigid", "de-novo", f"{_PACKAGE}.de_novo.frigid.inference"),
    Model("foam", "de-novo", f"{_PACKAGE}.de_novo.foam.inference"),
)}


def get_model(name: str) -> Model:
    return MODELS[name]
