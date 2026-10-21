import pytest

from msms_models.evaluation.metrics import MetricProtocol, evaluate, score_one


def test_absent_predictions_keep_the_manifest_denominator():
    queries = [{"id": "a", "target_smiles": "CCO"}, {"id": "b", "target_smiles": "CCN"}]
    summary, details = evaluate(
        queries,
        [{"id": "a", "candidates": ["CCO"]}],
        MetricProtocol(bootstrap_resamples=0),
    )
    assert summary["num_queries"] == 2
    assert summary["missing_predictions"] == 1
    assert summary["metrics"]["acc@1"]["mean"] == 0.5
    assert details[1]["status"] == "missing"


def test_invalid_slots_do_not_promote_later_predictions():
    p = MetricProtocol(bootstrap_resamples=0)
    row = score_one("CCO", [None, "CCO"], p)
    assert row["acc@1"] == 0
    assert row["acc@10"] == 1
    assert row["tanimoto@1"] == 0


def test_explicit_native_postprocessing_can_left_pack():
    p = MetricProtocol(candidate_policy="unique-valid", bootstrap_resamples=0)
    assert score_one("CCO", [None, "CCO"], p)["acc@1"] == 1


def test_connectivity_match_ignores_stereochemistry():
    p = MetricProtocol(bootstrap_resamples=0)
    row = score_one("C[C@H](O)C(=O)O", ["C[C@@H](O)C(=O)O"], p)
    assert row["acc@1"] == 1
    assert row["tanimoto@1"] == 1


def test_retrieval_only_solves_top_one_mces():
    calls = []

    def distance(a, b):
        calls.append((a, b))
        return 3.0

    p = MetricProtocol(task="retrieval", cutoffs=(1, 5, 20), mces=True)
    row = score_one("CCO", ["CCN", "CCO"], p, distance)
    assert calls == [("CCO", "CCN")]
    assert row["hit@1"] == 0 and row["hit@5"] == 1
    assert row["mces@1"] == 3
    assert "mces@20" not in row


def test_mces_solver_failures_are_not_silently_substituted():
    def broken(a, b):
        raise RuntimeError("solver failed")

    with pytest.raises(RuntimeError, match="solver failed"):
        score_one("CCO", ["CCN"], MetricProtocol(mces=True), broken)


def test_constant_bootstrap_has_finite_bounds():
    summary, _ = evaluate(
        [{"id": "a", "target_smiles": "CCO"}],
        [{"id": "a", "candidates": ["CCO"]}],
    )
    assert summary["metrics"]["acc@1"]["ci"] == [1, 1]


def test_mces_disk_cache_survives_new_evaluator(tmp_path, monkeypatch):
    import importlib

    pytest.importorskip("pulp")
    module = pytest.importorskip("myopic_mces")
    metrics = importlib.import_module("msms_models.evaluation.metrics")
    calls = []
    monkeypatch.setattr(module, "MCES", lambda *a, **kw: (calls.append((a, kw)), 1.5))
    protocol = metrics.MetricProtocol(mces=True, mces_cache=str(tmp_path / "pairs.sqlite"))
    assert metrics.make_mces(protocol)("CCO", "CCN") == 1.5
    assert metrics.make_mces(protocol)("CCO", "CCN") == 1.5
    assert len(calls) == 1
    assert calls[0][1]["solver_options"]["threads"] == 1


def test_tanimoto_preserves_explicit_hydrogen_input_graph():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    from msms_models.evaluation.chemistry import tanimoto2048

    target = "CCN"
    predicted = "[H]/N=C(\\N)SC"
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    expected = DataStructs.TanimotoSimilarity(
        generator.GetFingerprint(Chem.MolFromSmiles(target)),
        generator.GetFingerprint(Chem.MolFromSmiles(predicted)),
    )
    assert tanimoto2048(target, predicted) == expected


def test_real_mces_solver_on_a_known_one_bond_difference():
    import platform

    pulp = pytest.importorskip("pulp")
    pytest.importorskip("myopic_mces")
    from msms_models.evaluation.metrics import make_mces

    available = pulp.listSolvers(onlyAvailable=True)
    solver = "COIN_CMD" if "COIN_CMD" in available else "PULP_CBC_CMD"
    if solver not in available:
        pytest.skip("CBC is not installed")
    if solver == "PULP_CBC_CMD" and platform.system() == "Darwin":
        pytest.skip("Use native CBC on macOS; bundled executable may target another CPU")
    distance = make_mces(MetricProtocol(mces=True, mces_solver=solver))
    assert distance("CCO", "CCO") == 0.0
    assert distance("CCO", "CCCO") == 1.0


def test_invalid_and_missing_slots_have_mces_100_and_minima_are_capped():
    p = MetricProtocol(mces=True, bootstrap_resamples=0)
    row = score_one("CCO", ["not a smiles"], p, lambda a, b: 3.0)
    assert row["mces@1"] == 100 and row["mces@10"] == 100
    assert score_one("CCO", [], p, lambda a, b: 3.0)["mces@10"] == 100
    # A lower bound beyond the invalid distance is capped, as in MassSpecGym.
    assert score_one("CCO", ["CCN"], p, lambda a, b: 140.0)["mces@1"] == 100


def test_disconnected_predictions_are_valid_molecules():
    p = MetricProtocol(bootstrap_resamples=0)
    row = score_one("CCO", ["CCO.O"], p)
    assert row["acc@1"] == 0
    assert 0 < row["tanimoto@1"] < 1


def test_paper_protocols_are_fixed():
    from msms_models.evaluation.protocols import get_protocol

    table3, table5 = get_protocol("table3"), get_protocol("table5")
    assert (table3.task, table3.cutoffs, table3.mces) == ("retrieval", (1, 5, 20), True)
    assert (table5.task, table5.cutoffs, table5.candidate_policy) == ("de-novo", (1, 10), "preserve")
    assert table3.mces_threshold == table5.mces_threshold == 15
    assert get_protocol("table3", seed=1).seed == 1


def test_cli_protocol_with_explicit_overrides(tmp_path):
    import json

    from msms_models.cli import main
    from msms_models.data.io import write_jsonl

    write_jsonl(tmp_path / "q.jsonl", [{"id": "a", "target_smiles": "CCO"}])
    write_jsonl(tmp_path / "p.jsonl", [{"id": "a", "candidates": ["OCC"]}])
    main([
        "evaluate", "--protocol", "table5", "--queries", str(tmp_path / "q.jsonl"),
        "--predictions", str(tmp_path / "p.jsonl"), "--output", str(tmp_path / "m.json"),
        "--bootstrap-resamples", "0", "--seed", "42",
    ])
    summary = json.loads((tmp_path / "m.json").read_text())
    assert summary["protocol"]["cutoffs"] == [1, 10] and summary["protocol"]["seed"] == 42
    assert summary["metrics"]["acc@1"]["mean"] == 1 and summary["metrics"]["mces@1"]["mean"] == 0
