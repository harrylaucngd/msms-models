"""Command-line interface. Model runtimes are imported only by the command that needs them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="msms-models")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="List registered models and their tasks")
    download = sub.add_parser("download", help="Download the official MassSpecGym files")
    download.add_argument("dataset", choices=["msg", "massspecgym"])
    download.add_argument("--output", type=Path, default=Path("data"))
    download.add_argument("--candidates", choices=["formula", "mass", "both"])
    download.add_argument("--candidate-version", choices=["original", "1.5"], default="original")
    download.add_argument("--molecules", action="store_true",
                          help="Also the 4M molecules that train the SMILES Transformer tokenizer")
    candidates = sub.add_parser(
        "prepare-candidates", help="Convert official candidates to ordered spectrum-ID records"
    )
    candidates.add_argument("--input", type=Path, required=True)
    candidates.add_argument("--queries", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser(
        "prepare", help="Normalize tabular, JSONL or annotated parquet inputs into query records"
    )
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--split", choices=["train", "val", "test"])
    prepare.add_argument(
        "--formula-from-target",
        action="store_true",
        help="For formula-known benchmarks, derive molecular formula from target SMILES",
    )
    predict = sub.add_parser("predict", help="Run a configured model and record its predictions")
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--queries", type=Path, help="Override the configured query file")
    predict.add_argument("--output", type=Path, help="Override the output directory")
    predict.add_argument("--device", help="Override the device, e.g. cuda:0")
    predict.add_argument("--limit", type=int, help="Predict only the first N queries")
    fit = sub.add_parser(
        "fit-nn", help="Build a train-only neighbour index and select parameters on validation"
    )
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate predictions against a query file")
    evaluate.add_argument("--queries", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    task = evaluate.add_mutually_exclusive_group(required=True)
    task.add_argument("--protocol", choices=["table2", "table3", "table5"], help="A fixed paper protocol")
    task.add_argument("--task", choices=["de-novo", "retrieval"])
    # Unset options keep the protocol (or task) defaults.
    evaluate.add_argument("--cutoffs", nargs="+", type=int)
    evaluate.add_argument("--candidate-policy", choices=["preserve", "unique-valid"])
    evaluate.add_argument("--canonical-tautomers", dest="tautomer", action="store_const", const=True)
    evaluate.add_argument("--mces", action="store_const", const=True)
    evaluate.add_argument(
        "--mces-solver", help="PuLP solver name; use COIN_CMD for a native CBC executable on PATH"
    )
    evaluate.add_argument("--mces-time-limit", type=float)
    evaluate.add_argument("--mces-cache", help="Resume completed MCES pairs from a SQLite file")
    evaluate.add_argument("--mces-threads", type=int)
    evaluate.add_argument("--bootstrap-resamples", type=int)
    evaluate.add_argument("--confidence", type=float)
    evaluate.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.command == "models":
        from .models import MODELS

        print(json.dumps([m.to_dict() for m in MODELS.values()], indent=2))
    elif args.command == "download":
        from .data.download import download_massspecgym

        print(
            json.dumps(
                download_massspecgym(
                    args.output,
                    candidates=args.candidates,
                    candidate_version=args.candidate_version,
                    molecules=args.molecules,
                ),
                indent=2,
            )
        )
    elif args.command == "prepare-candidates":
        from .data.prepare import prepare_candidates

        print(json.dumps(prepare_candidates(args.input, args.queries, args.output), indent=2))
    elif args.command == "prepare":
        from .data.prepare import prepare_queries

        print(
            json.dumps(
                prepare_queries(
                    args.input,
                    args.output,
                    split=args.split,
                    formula_from_target=args.formula_from_target,
                ),
                indent=2,
            )
        )
    elif args.command == "predict":
        from dataclasses import replace

        from .api import run
        from .config import RunConfig

        config = RunConfig.from_file(args.config)
        overrides = {}
        for key in ("queries", "output", "device"):
            if (value := getattr(args, key)) is not None:
                overrides[key] = value.resolve() if isinstance(value, Path) else value
        print(json.dumps(run(replace(config, **overrides), limit=args.limit), indent=2))
    elif args.command == "fit-nn":
        from .data.io import load_queries
        from .models.retrieval.nearest_neighbor.index import fit

        selection = fit(load_queries(args.train), load_queries(args.validation), args.output)
        print(json.dumps(selection, indent=2))
    else:
        from dataclasses import fields, replace

        from .data.io import load_queries, read_records, write_json, write_jsonl
        from .evaluation.metrics import MetricProtocol
        from .evaluation.metrics import evaluate as run
        from .evaluation.protocols import get_protocol

        if args.protocol:
            protocol = get_protocol(args.protocol)
        else:
            cutoffs = (1, 5, 20) if args.task == "retrieval" else (1, 10)
            protocol = MetricProtocol(task=args.task, cutoffs=cutoffs)
        overrides = {
            f.name: getattr(args, f.name) for f in fields(MetricProtocol)
            if getattr(args, f.name, None) is not None and f.name != "task"
        }
        if "cutoffs" in overrides:
            overrides["cutoffs"] = tuple(overrides["cutoffs"])
        protocol = replace(protocol, **overrides)
        summary, details = run(
            load_queries(args.queries),
            read_records(args.predictions),
            protocol,
        )
        write_json(args.output, summary)
        write_jsonl(args.output.with_suffix(".details.jsonl"), details)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
