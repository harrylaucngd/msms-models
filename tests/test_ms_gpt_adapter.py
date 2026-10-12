import json
from dataclasses import replace

import pytest

from msms_models.backends.ms_gpt_adapter import (
    merge_adapter,
    validate_adapter_assets,
    validate_adapter_training,
)
from msms_models.config import RunConfig
from msms_models.validation import required_assets, validate_run


def test_two_models_have_distinct_adapter_contracts(tmp_path):
    base = RunConfig(
        model="ms-gpt-base", dataset="msg", queries=tmp_path / "q", output=tmp_path / "out"
    )
    assert "adapter_checkpoint" not in required_assets(base)
    assert "adapter_checkpoint" in required_assets(replace(base, model="ms-gpt"))
    with pytest.raises(ValueError, match="must not receive"):
        validate_run(replace(base, assets={"adapter_checkpoint": str(tmp_path)}), [])


def test_adapter_config_alone_is_not_a_trained_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    with pytest.raises(FileNotFoundError, match="no adapter_model"):
        validate_adapter_assets(tmp_path)


def test_adapter_provenance_rejects_incompatible_calibration(tmp_path):
    path = tmp_path / "training_config.json"
    path.write_text(json.dumps({"benchmark": "nplib1", "training_split": "train"}))
    with pytest.raises(ValueError, match="different benchmark"):
        validate_adapter_training(tmp_path, "msg", {})
    path.write_text(json.dumps({"benchmark": "massspecgym", "training_split": "test"}))
    with pytest.raises(ValueError, match="training split"):
        validate_adapter_training(tmp_path, "msg", {})
    path.write_text(
        json.dumps({"benchmark": "massspecgym", "threshold_low": 0.1, "threshold_high": 0.3})
    )
    with pytest.raises(ValueError, match="different threshold bands"):
        validate_adapter_training(tmp_path, "msg", {"threshold_low": 0.1, "threshold_high": 0.4})


def _small_model():
    from msms_models._vendor.ms_gpt.model import ConditionalMoleculeLanguageModel

    return ConditionalMoleculeLanguageModel(
        vocab_size=98,
        d_model=32,
        n_layers=2,
        n_heads=4,
        fp_bits=4096,
        fp_encoder_layers=1,
        cross_attn_every_n=1,
        max_seq_len=16,
        dropout=0.0,
    ).eval()


def _adapter_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = _small_model()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wrapped = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["context_proj"], bias="none"
        ),
    )
    with torch.no_grad():
        for name, p in wrapped.named_parameters():
            if "lora_" in name:
                p.normal_(mean=0.0, std=0.05)
    wrapped.eval().save_pretrained(tmp_path)
    return torch, wrapped, original


def test_nonzero_adapter_merge_matches_peft_and_changes_base(tmp_path):
    torch, reference, state = _adapter_fixture(tmp_path)
    base = _small_model()
    base.load_state_dict(state)
    merged = merge_adapter(base, tmp_path)
    x = torch.randn(2, 3, 32)
    with torch.inference_mode():
        expected = reference.base_model.model.context_proj(x)
        actual = merged.context_proj(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not torch.equal(merged.context_proj.weight, state["context_proj.weight"])


def test_adapter_missing_learned_tensor_fails(tmp_path):
    torch, _, state = _adapter_fixture(tmp_path)
    from safetensors.torch import load_file, save_file

    path = tmp_path / "adapter_model.safetensors"
    tensors = load_file(path)
    tensors.pop(next(iter(tensors)))
    save_file(tensors, path)
    base = _small_model()
    base.load_state_dict(state)
    with pytest.raises(ValueError, match="tensor keys"):
        merge_adapter(base, tmp_path)
