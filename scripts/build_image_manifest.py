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

from data_merge.image_manifest import ImageManifestConfig, build_image_manifest


DEFAULT_DATASET_ROOT = "/workspace/data_dir/data_user/public_data/360video/outdoor/test"
DEFAULT_OUTPUT_MANIFEST = "/workspace/data_dir/data_user/public_data/360video/outdoor/test/image_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an image manifest JSON for an ERP image benchmark directory.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Benchmark image dataset root.")
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST, help="Output JSON manifest path.")
    parser.add_argument("--dataset-name", default="real_360_test", help="Dataset name written into the manifest.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker count.")
    parser.add_argument("--resize-width", type=int, default=2048, help="Output image width.")
    parser.add_argument("--resize-height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N processed images.")
    args = parser.parse_args()

    payload = build_image_manifest(
        ImageManifestConfig(
            dataset_root=Path(args.dataset_root),
            output_manifest_path=Path(args.output_manifest),
            dataset_name=args.dataset_name,
            workers=args.workers,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
            progress_every=args.progress_every,
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
