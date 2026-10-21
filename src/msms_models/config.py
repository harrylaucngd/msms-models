"""Run configurations; paths are relative to the configuration file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    model: str
    dataset: str
    queries: Path
    output: Path
    assets: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    device: str = "cpu"
    workers: int = 1
    seed: int = 0

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text())

        def resolve(value):
            value = Path(value).expanduser()
            return value if value.is_absolute() else (path.parent / value).resolve()

        return cls(
            model=raw["model"],
            dataset=raw["dataset"],
            queries=resolve(raw["queries"]),
            output=resolve(raw.get("output", "results")),
            assets={key: str(resolve(value)) for key, value in raw.get("assets", {}).items()},
            parameters=raw.get("parameters", {}),
            device=str(raw.get("device", "cpu")),
            workers=int(raw.get("workers", 1)),
            seed=int(raw.get("seed", 0)),
        )

    def require_assets(self, *names: str) -> dict[str, Path]:
        return {name: Path(self.assets[name]) for name in names}
