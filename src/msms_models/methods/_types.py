"""Dependency-free descriptions of runnable methods and their shared contracts."""

from dataclasses import asdict, dataclass
from importlib import import_module


@dataclass(frozen=True)
class Model:
    name: str
    task: str
    datasets: tuple[str, ...]
    formula_required: bool
    backend: str  # Kept for the 0.2 Python API.
    description: str
    upstream: str
    method: str
    module: str
    configuration: str
    assets: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    uses_mist: bool = False

    def to_dict(self):
        return asdict(self)

    def config_for(self, dataset):
        """Return the checkout-relative example configuration for this variant."""
        if dataset not in self.datasets:
            raise ValueError(f"{self.name} does not support dataset {dataset!r}")
        return self.configuration.format(dataset=dataset)

    def validate(self, config, queries):
        """Apply method-specific input checks without constructing a model."""
        specification = import_module(f"msms_models.methods.{self.method}.spec")
        validator = getattr(specification, "validate", None)
        if validator is not None:
            validator(config, queries)

    def predict(self, config, queries):
        """Dispatch to the method implementation in the current interpreter."""
        return import_module(self.module).predict(config, queries)

    def required_assets(self, config):
        assets = list(self.assets)
        if self.uses_mist:
            assets += (
                ["posterior", "posterior_ids", "posterior_manifest"]
                if "posterior" in config.assets
                else ["mist_checkpoint",
                      "legacy_features" if "legacy_features" in config.assets else "compact_cache"]
            )
        return assets

    def required_modules(self, config):
        modules = {"numpy", "scipy", "rdkit", *self.dependencies}
        if self.uses_mist and "posterior" not in config.assets:
            modules.update(["torch"])
            modules.update(["pyarrow", "datamol"] if "legacy_features" in config.assets else ["mist"])
        return sorted(modules)
