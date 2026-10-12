"""Public command-line interface. Optional models never import during --help."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="msms-models")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="List model/task/input contracts")
    doctor = sub.add_parser(
        "doctor", help="Check input assets and backend dependencies without loading weights"
    )
    doctor.add_argument("--config", type=Path, required=True)
    prepare = sub.add_parser(
        "prepare", help="Normalize TSV, CSV or JSONL into an ordered query manifest"
    )
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--split", choices=["train", "val", "test"])
    prepare.add_argument(
        "--formula-from-target",
        action="store_true",
        help="For formula-known benchmarks, derive molecular formula from target SMILES",
    )
    infer = sub.add_parser("infer", help="Run the configured native inference backend")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--dry-run", action="store_true")
    fit = sub.add_parser(
        "fit-nn", help="Build a train-only neighbour index and select parameters on validation"
    )
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--dataset", choices=["msg"], default="msg")
    evaluate = sub.add_parser("evaluate", help="Evaluate against a fixed query universe")
    evaluate.add_argument("--queries", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--task", choices=["de-novo", "retrieval"], required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--cutoffs", nargs="+", type=int)
    evaluate.add_argument(
        "--candidate-policy", choices=["preserve", "unique-valid"], default="preserve"
    )
    evaluate.add_argument("--canonical-tautomers", action="store_true")
    evaluate.add_argument("--mces", action="store_true")
    evaluate.add_argument("--mces-time-limit", type=float, default=5.0)
    evaluate.add_argument(
        "--mces-cache", type=str, help="Resume completed MCES pairs from a SQLite file"
    )
    evaluate.add_argument("--mces-threads", type=int, default=1)
    evaluate.add_argument("--bootstrap-resamples", type=int, default=20000)
    evaluate.add_argument("--confidence", type=float, default=0.999)
    evaluate.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "models":
        from .registry import MODELS

        print(json.dumps([m.to_dict() for m in MODELS.values()], indent=2))
    elif args.command == "doctor":
        from .config import RunConfig
        from .io import load_queries
        from .validation import doctor as check

        config = RunConfig.from_file(args.config)
        print(json.dumps(check(config, load_queries(config.queries)), indent=2))
    elif args.command == "prepare":
        from .io import load_queries, write_jsonl

        records = load_queries(args.input, split=args.split)
        if args.formula_from_target:
            from rdkit import Chem
            from rdkit.Chem.rdMolDescriptors import CalcMolFormula

            for record in records:
                mol = Chem.MolFromSmiles(record.get("target_smiles", ""))
                if mol is None or not mol.GetNumAtoms():
                    raise ValueError(f"Cannot derive formula for {record['id']}")
                formula = CalcMolFormula(mol)
                if record.get("formula") and record["formula"] != formula:
                    raise ValueError(
                        f"Provided formula differs from target structure for {record['id']}"
                    )
                record["formula"] = formula
        write_jsonl(args.output, records)
        print(json.dumps({"queries": len(records), "output": str(args.output)}))
    elif args.command == "infer":
        from .api import infer as run
        from .config import RunConfig

        print(json.dumps(run(RunConfig.from_file(args.config), dry_run=args.dry_run), indent=2))
    elif args.command == "fit-nn":
        from .io import load_queries
        from .retrieval.subformula import fit

        selection = fit(
            load_queries(args.train, require_target=True),
            load_queries(args.validation, require_target=True),
            args.output,
            dataset=args.dataset,
        )
        print(json.dumps(selection, indent=2))
    else:
        from .io import atomic_json, load_queries, read_records, write_jsonl
        from .metrics import MetricProtocol
        from .metrics import evaluate as run

        cutoffs = tuple(args.cutoffs or ([1, 5, 20] if args.task == "retrieval" else [1, 10]))
        protocol = MetricProtocol(
            task=args.task,
            cutoffs=cutoffs,
            candidate_policy=args.candidate_policy,
            tautomer=args.canonical_tautomers,
            mces=args.mces,
            mces_time_limit=args.mces_time_limit,
            bootstrap_resamples=args.bootstrap_resamples,
            mces_cache=args.mces_cache,
            mces_threads=args.mces_threads,
            confidence=args.confidence,
            seed=args.seed,
        )
        summary, details = run(
            load_queries(args.queries, require_target=True),
            read_records(args.predictions),
            protocol,
        )
        atomic_json(args.output, summary)
        write_jsonl(args.output.with_suffix(".details.jsonl"), details)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
