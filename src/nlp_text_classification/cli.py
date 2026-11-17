"""Simple reproducible CLI; later experiments extend the same command surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="nlp-text-classification")
    command = parser.add_subparsers(dest="command", required=True)
    dl = command.add_parser("download", help="Download official GoEmotions simplified split")
    dl.add_argument("--data-dir", default="data/goemotions")
    dl.add_argument("--force", action="store_true")
    check = command.add_parser("validate", help="Validate split and label contract")
    check.add_argument("--data-dir", default="data/goemotions")
    e1 = command.add_parser("train-e1", help="Train the E1 SVM baseline")
    e1.add_argument("--data-dir", default="data/goemotions")
    e1.add_argument("--output-root", default="artifacts")
    e1.add_argument("--seed", type=int, default=42)
    e1.add_argument("--max-features", type=int, default=100_000)
    e1.add_argument("--min-df", type=int, default=2)
    final = command.add_parser("evaluate-final-e1", help="One-time test of frozen E1 run")
    final.add_argument("--data-dir", default="data/goemotions")
    final.add_argument("--run-dir", required=True)
    pred = command.add_parser("predict", help="Infer using an E1 trained run")
    pred.add_argument("--run-dir", required=True)
    pred.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "download":
        from nlp_text_classification.data.download import download_official
        result = download_official(args.data_dir, force=args.force)
    elif args.command == "validate":
        from nlp_text_classification.data.goemotions import validate_dataset
        result = validate_dataset(args.data_dir)
    elif args.command == "train-e1":
        from nlp_text_classification.experiments.e1_svm import train
        run_dir = train(args.data_dir, args.output_root, seed=args.seed,
                        max_features=args.max_features,
                        min_df=args.min_df)
        result = {"run_dir": str(run_dir)}
    elif args.command == "evaluate-final-e1":
        from nlp_text_classification.experiments.e1_svm import evaluate_final
        result = evaluate_final(args.run_dir, args.data_dir)
    else:
        from nlp_text_classification.experiments.e1_svm import predict
        result = predict(args.run_dir, [args.text])[0]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
