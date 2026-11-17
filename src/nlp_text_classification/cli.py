"""Dataset audit, standalone E1 training, and local inference commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlp_text_classification.data.loaders import goemotions, movie_reviews_zh


def main() -> None:
    parser = argparse.ArgumentParser(prog="nlp-text-classification")
    command = parser.add_subparsers(dest="command", required=True)
    dl = command.add_parser("download", help="Download official GoEmotions simplified split")
    dl.add_argument("--data-dir", type=Path, default=goemotions.DEFAULT_ROOT)
    dl.add_argument("--force", action="store_true")
    check = command.add_parser("validate", help="Audit all GoEmotions source files")
    check.add_argument("--data-dir", type=Path, default=goemotions.DEFAULT_ROOT)
    audit = command.add_parser("audit-data", help="Audit all splits without fitting a model")
    audit.add_argument("--dataset", choices=("goemotions", "movie_reviews_zh"), required=True)
    audit.add_argument("--data-dir", type=Path)
    freeze = command.add_parser("freeze-data", help="Create audited immutable experiment splits")
    freeze.add_argument("--dataset", choices=("goemotions", "movie_reviews_zh"), required=True)
    freeze.add_argument("--data-dir", type=Path)
    freeze.add_argument("--output", type=Path)
    freeze.add_argument("--revision", required=True)
    freeze.add_argument("--positive-id", type=int, choices=(0, 1))
    e1 = command.add_parser("train-e1", help="Train and validate the standalone GoEmotions E1")
    e1.add_argument("--data-dir", type=Path, default=goemotions.DEFAULT_ROOT)
    e1.add_argument("--output-root", default="artifacts")
    e1.add_argument("--seed", type=int, default=42)
    e1.add_argument("--max-features", type=int, default=100_000)
    e1.add_argument("--min-df", type=int, default=2)
    pred = command.add_parser("predict", help="Infer using an E1 trained run")
    pred.add_argument("--run-dir", required=True)
    pred.add_argument("--text", required=True)
    batch = command.add_parser("batch-a", help="Run Batch A with logged progress")
    batch.add_argument(
        "mode", choices=("smoke", "run", "audit", "final-test", "report", "status", "resources")
    )
    batch.add_argument("--suite", type=Path, default=Path("configs/experiment_suites/batch_a.yaml"))
    batch.add_argument("--output", type=Path, default=Path("artifacts/batch-a"))
    batch.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "download":
        from nlp_text_classification.data.download import download_official
        result = download_official(args.data_dir, force=args.force)
    elif args.command == "validate":
        result = goemotions.validate_dataset(args.data_dir)
    elif args.command == "audit-data":
        if args.dataset == "goemotions":
            result = goemotions.validate_dataset(args.data_dir or goemotions.DEFAULT_ROOT)
        else:
            result = movie_reviews_zh.audit_dataset(args.data_dir or movie_reviews_zh.DEFAULT_ROOT)
    elif args.command == "freeze-data":
        from nlp_text_classification.data.freeze import freeze_dataset

        default_root = (goemotions.DEFAULT_ROOT if args.dataset == "goemotions"
                        else movie_reviews_zh.DEFAULT_ROOT)
        result = freeze_dataset(
            args.dataset, args.data_dir or default_root,
            args.output or (Path("data/processed") / args.dataset),
            args.revision, positive_id=args.positive_id,
        )
    elif args.command == "train-e1":
        from nlp_text_classification.experiments.e1_svm import train
        run_dir = train(args.data_dir, args.output_root, seed=args.seed,
                        max_features=args.max_features,
                        min_df=args.min_df)
        result = {"run_dir": str(run_dir)}
    elif args.command == "batch-a":
        if args.mode == "status":
            snapshot = args.output / "progress.md"
            result = {"progress": str(snapshot), "text": snapshot.read_text(encoding="utf-8")}
        elif args.mode == "resources":
            from nlp_text_classification.training.resources import resource_status

            result = resource_status()
        else:
            from nlp_text_classification.training.resources import (
                configure_compute_threads,
                require_headroom,
            )

            try:
                require_headroom()
            except RuntimeError as exc:
                parser.exit(status=75, message=f"{exc}\n")
            configure_compute_threads()
            from nlp_text_classification.experiments.batch_cli import batch_command

            result = batch_command(args.mode, args.suite, args.output, args.device)
    else:
        from nlp_text_classification.experiments.e1_svm import predict
        result = predict(args.run_dir, [args.text])[0]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
