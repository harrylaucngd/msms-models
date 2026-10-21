"""Validate and merge the MS-GPT posterior-aligned PEFT LoRA adapter."""

from __future__ import annotations

import json
from pathlib import Path


def validate_adapter_assets(path):
    path = Path(path)
    if not path.is_dir() or not (path / "adapter_config.json").is_file():
        raise ValueError(
            "adapter_checkpoint must be a PEFT directory containing adapter_config.json"
        )
    if not any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(
            "The MS-GPT adapter directory has no adapter_model.safetensors or adapter_model.bin"
        )
    metadata = json.loads((path / "adapter_config.json").read_text())
    if metadata.get("peft_type") != "LORA":
        raise ValueError("MS-GPT requires a posterior-aligned LoRA adapter")
    return path


def validate_adapter_training(path, dataset, calibration):
    """Honor training provenance when supplied by the adapter publisher."""
    from ....config import dataset_name

    path = Path(path) / "training_config.json"
    if not path.exists():
        return
    metadata = json.loads(path.read_text())
    if metadata.get("benchmark") and dataset_name(metadata["benchmark"]) != dataset:
        raise ValueError("MS-GPT adapter was trained for a different benchmark")
    if metadata.get("training_split", "train") != "train":
        raise ValueError("MS-GPT adapter must be trained on the training split")
    for key in ("threshold_low", "threshold_high"):
        if key in metadata and abs(float(metadata[key]) - float(calibration[key])) > 1e-8:
            raise ValueError("MS-GPT adapter and density calibration use different threshold bands")


def merge_adapter(model, path):
    """Match upstream PEFT inference, rejecting missing or ignored learned tensors."""
    import torch
    from peft import LoraConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict, load_peft_weights

    path = validate_adapter_assets(path)
    weights = load_peft_weights(str(path), device="cpu", local_files_only=True)
    if not weights:
        raise ValueError("MS-GPT adapter weights are empty")
    for name, tensor in weights.items():
        module = name.removeprefix("base_model.model.")
        if (
            not module.startswith(("fp_encoder.", "context_proj.", "cross_attn_layers."))
            or ".lora_" not in module
        ):
            raise ValueError(f"Adapter tensor is outside the posterior-reading pathway: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite adapter tensor: {name}")
    metadata = json.loads((path / "adapter_config.json").read_text())
    # The authorized Hub mirror labels this custom architecture as "CUSTOM".
    # PEFT represents a generic PeftModel with task_type=None, not that string.
    if metadata.get("task_type") == "CUSTOM":
        metadata["task_type"] = None
    adapter_config = LoraConfig(**metadata)
    wrapped = PeftModel.from_pretrained(
        model, str(path), config=adapter_config, is_trainable=False, local_files_only=True
    )
    loaded = get_peft_model_state_dict(wrapped)
    if set(loaded) != set(weights):
        raise ValueError("Adapter tensor keys do not exactly match the configured LoRA modules")
    for name, expected in weights.items():
        actual = loaded[name].detach().cpu()
        if actual.shape != expected.shape or not torch.equal(actual, expected.to(actual.dtype)):
            raise ValueError(f"MS-GPT adapter tensor was not loaded exactly: {name}")
    return wrapped.merge_and_unload(safe_merge=True).eval()
