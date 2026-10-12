"""Small dependency records without importing optional model packages."""

from importlib.metadata import PackageNotFoundError, version
from platform import python_version

from . import __version__


def environment_record():
    result = {"python": python_version(), "msms-models": __version__}
    for name in (
        "numpy",
        "scipy",
        "rdkit",
        "torch",
        "mist",
        "transformers",
        "dgl",
        "ms-pred",
        "sentencepiece",
        "bionemo-moco",
        "joblib",
        "torch-scatter",
        "pytorch-lightning",
        "lightning",
        "safe-mol",
        "ngboost",
        "scikit-learn",
        "pymoo",
        "myopic-mces",
        "pulp",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result
