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

from data_merge.builder import DEFAULT_WORKERS, BuildConfig, build_dataset
from data_merge.config import DEFAULT_CAPTION_JSONL, DEFAULT_GROUNDING_JSON, DEFAULT_RESULTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge panorama QA, caption, and grounding data into one SFT corpus.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Path to results_final_v2.")
    parser.add_argument("--caption-jsonl", default=DEFAULT_CAPTION_JSONL, help="Path to output_v.jsonl.")
    parser.add_argument("--grounding-json", default=DEFAULT_GROUNDING_JSON, help="Path to pano_grounding_train_factory.json.")
    parser.add_argument("--output-dir", help="Directory for merged outputs.")
    parser.add_argument("--use-mock-data", action="store_true", help="Use the bundled mock inputs instead of server paths.")
    parser.add_argument("--no-full-caption", action="store_true", help="Do not emit the full-caption sample for each caption record.")
    parser.add_argument("--drop-missing-images", action="store_true", help="Skip samples that do not have usable images.")
    parser.add_argument("--workers", type=int, default=None, help="Thread worker count for parallel normalization.")
    parser.add_argument("--max-results-records", type=int, help="Optional cap for normalized results_final_v2 records.")
    parser.add_argument("--max-caption-records", type=int, help="Optional cap for caption-derived SFT records.")
    parser.add_argument("--max-grounding-records", type=int, help="Optional cap for grounding records.")
    args = parser.parse_args()

    if args.use_mock_data:
        mock_root = ROOT / "examples" / "mock_input"
        results_dir = mock_root / "results_final_v2"
        caption_jsonl = mock_root / "vqa_generation" / "output_v.jsonl"
        grounding_json = mock_root / "grounding" / "pano_grounding_train_factory.json"
        output_dir = Path(args.output_dir) if args.output_dir else ROOT / "examples" / "mock_output"
    else:
        results_dir = Path(args.results_dir)
        caption_jsonl = Path(args.caption_jsonl)
        grounding_json = Path(args.grounding_json)
        output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "sft_merge"

    config = BuildConfig(
        results_dir=results_dir,
        caption_jsonl=caption_jsonl,
        grounding_json=grounding_json,
        output_dir=output_dir,
        include_full_caption=not args.no_full_caption,
        drop_missing_images=args.drop_missing_images,
        workers=args.workers if args.workers is not None else DEFAULT_WORKERS,
        max_results_records=args.max_results_records,
        max_caption_records=args.max_caption_records,
        max_grounding_records=args.max_grounding_records,
    )
    stats = build_dataset(config)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
