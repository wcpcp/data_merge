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

from data_merge.video_frames import VideoFrameExtractionConfig, VideoSource, extract_uniform_video_frames


DEFAULT_SPHERE360_DIR = "/workspace/data_dir/data_user/public_data/360video/Sphere360/videos"
DEFAULT_PANOWAN_DIR = "/workspace/data_dir/data_user/public_data/360video/panowan"
DEFAULT_OUTPUT_IMAGES_ROOT = "/workspace/data_dir/data_user/public_data/360video/outdoor/images"
DEFAULT_OUTPUT_MANIFEST = "/workspace/data_dir/data_user/public_data/360video/outdoor/frame_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Uniformly extract ERP video frames and build a manifest JSON.")
    parser.add_argument("--sphere360-dir", default=DEFAULT_SPHERE360_DIR, help="Sphere360 source video directory.")
    parser.add_argument("--panowan-dir", default=DEFAULT_PANOWAN_DIR, help="Panowan source video directory.")
    parser.add_argument("--output-images-root", default=DEFAULT_OUTPUT_IMAGES_ROOT, help="Directory to store extracted images.")
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST, help="Final JSON manifest path.")
    parser.add_argument("--frames-per-video", type=int, default=5, help="Number of uniformly sampled frames per video.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker count.")
    parser.add_argument("--resize-width", type=int, default=2048, help="Output image width.")
    parser.add_argument("--resize-height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract frames even if outputs already exist.")
    args = parser.parse_args()

    config = VideoFrameExtractionConfig(
        sources=[
            VideoSource(name="Sphere360", root=Path(args.sphere360_dir)),
            VideoSource(name="panowan", root=Path(args.panowan_dir)),
        ],
        output_images_root=Path(args.output_images_root),
        output_manifest_path=Path(args.output_manifest),
        frames_per_video=args.frames_per_video,
        workers=args.workers,
        overwrite=args.overwrite,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )
    payload = extract_uniform_video_frames(config)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
