"""Paths are interpreted relative to the config, never an unrelated working directory."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .registry import dataset_name, get_model


@dataclass(frozen=True)
class RunConfig:
    model: str
    dataset: str
    queries: Path
    output: Path
    assets: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    python: str = sys.executable
    device: str = "cpu"
    workers: int = 1
    seed: int = 0

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Inference configuration must be a mapping")
        allowed = {
            "model",
            "dataset",
            "queries",
            "output",
            "assets",
            "parameters",
            "python",
            "device",
            "workers",
            "seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        base = path.parent

        def resolve(value):
            p = Path(value).expanduser()
            return p if p.is_absolute() else (base / p).resolve()

        model = get_model(raw["model"])
        dataset = dataset_name(raw["dataset"])
        if dataset not in model.datasets:
            raise ValueError(f"{model.name} does not support dataset {dataset!r}")
        workers = int(raw.get("workers", 1))
        if workers < 1:
            raise ValueError("workers must be at least one")
        assets = raw.get("assets", {})
        if not isinstance(assets, dict) or not isinstance(raw.get("parameters", {}), dict):
            raise ValueError("assets and parameters must be mappings")
        return cls(
            model=model.name,
            dataset=dataset,
            queries=resolve(raw["queries"]),
            output=resolve(raw.get("output", "results")),
            assets={key: str(resolve(value)) for key, value in assets.items() if value is not None},
            parameters=raw.get("parameters", {}),
            python=str(resolve(raw["python"]))
            if "python" in raw and "/" in raw["python"]
            else raw.get("python", sys.executable),
            device=str(raw.get("device", "cpu")),
            workers=workers,
            seed=int(raw.get("seed", 0)),
        )

    def require_assets(self, *names: str) -> dict[str, Path]:
        missing = [name for name in names if not self.assets.get(name)]
        if missing:
            raise ValueError(f"{self.model} requires assets: {', '.join(missing)}")
        assets = {name: Path(self.assets[name]) for name in names}
        for name, path in assets.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name}: {path}")
        return assets
