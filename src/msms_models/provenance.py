"""Small dependency records without importing optional model packages."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version

from . import __version__


def input_record(config):
    """Hash files; for directories identify manifests without implying a full-tree hash."""
    import hashlib

    from .datasets.download import sha256

    assets = {}
    for key, value in config.assets.items():
        path = Path(value)
        if path.is_file():
            assets[key] = {"sha256": sha256(path), "size": path.stat().st_size}
        elif path.is_dir():
            manifests = sorted(path.rglob("*manifest*.json"))
            assets[key] = {
                "directory_content_hashed": False,
                "manifests": {str(p.relative_to(path)): sha256(p) for p in manifests},
            }
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "queries_sha256": sha256(config.queries),
        "package_source_sha256": digest.hexdigest(),
        "assets": assets,
    }


def environment_record():
    result = {"python": python_version(), "msms-models": __version__}
    for name in (
        "numpy",
        "scipy",
        "rdkit",
        "torch",
        "mist",
        "transformers",
        "peft",
        "safetensors",
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
