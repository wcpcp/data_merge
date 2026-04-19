#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_merge.external_benchmark_sft import ExternalBenchmarkBuildConfig, build_external_benchmark_training_sets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect official training-capable data from OSR-Bench, Thinking in 360, and PanoEnv, "
            "then export three multimodal training JSON files."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "external_benchmark_sft"),
        help="Directory where the three training JSON files and stats file will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "outputs" / "external_benchmark_cache"),
        help="Directory used to cache downloaded source files and extracted images.",
    )
    parser.add_argument("--max-osr-records", type=int, help="Optional cap for OSR-Bench converted records.")
    parser.add_argument("--max-thinking-records", type=int, help="Optional cap for Thinking in 360 converted records.")
    parser.add_argument("--max-panoenv-records", type=int, help="Optional cap for PanoEnv converted records.")
    parser.add_argument(
        "--panoenv-question-types",
        default="open_ended",
        help="Comma-separated PanoEnv question_type filter. Default keeps only open_ended.",
    )
    parser.add_argument(
        "--include-osr-bench",
        action="store_true",
        help="Explicitly export OSR-Bench into training format. This is off by default because the public release is benchmark-oriented and does not provide an official QA train split.",
    )
    parser.add_argument(
        "--skip-thinking-in-360",
        action="store_true",
        help="Do not export the official Thinking in 360 panorama SFT data.",
    )
    parser.add_argument("--skip-panoenv", action="store_true", help="Do not export PanoEnv.")
    args = parser.parse_args()

    config = ExternalBenchmarkBuildConfig(
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        max_osr_records=args.max_osr_records,
        max_thinking_records=args.max_thinking_records,
        max_panoenv_records=args.max_panoenv_records,
        include_osr_bench=args.include_osr_bench,
        include_thinking_in_360=not args.skip_thinking_in_360,
        include_panoenv=not args.skip_panoenv,
        panoenv_allowed_question_types=tuple(
            value.strip() for value in args.panoenv_question_types.split(",") if value.strip()
        ),
    )
    stats = build_external_benchmark_training_sets(config)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
