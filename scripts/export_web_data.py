"""CLI entrypoint to regenerate static dashboard data JSON from frozen experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_text_classification.evaluation.export_web_data import export_web_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen EmotionLab metrics for apps/web.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/web/src/data/dashboard-data.json"),
        help="Destination JSON file path",
    )
    args = parser.parse_args()
    out = export_web_data(root=args.root, output_path=args.output)
    print(f"Exported dashboard data to {out}")


if __name__ == "__main__":
    main()
