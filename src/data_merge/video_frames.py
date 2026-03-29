from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


COMMON_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".ts",
    ".mts",
    ".m2ts",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
}


@dataclass
class VideoSource:
    name: str
    root: Path


@dataclass
class VideoFrameExtractionConfig:
    sources: List[VideoSource]
    output_images_root: Path
    output_manifest_path: Path
    frames_per_video: int = 5
    workers: int = min(16, (os.cpu_count() or 4))
    overwrite: bool = False
    image_extension: str = ".jpg"


def extract_uniform_video_frames(config: VideoFrameExtractionConfig) -> Dict[str, Any]:
    video_jobs = discover_video_jobs(config.sources)
    config.output_images_root.mkdir(parents=True, exist_ok=True)
    config.output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = parallel_map(
        video_jobs,
        lambda job: process_video_job(
            dataset_name=job[0],
            source_root=job[1],
            video_path=job[2],
            config=config,
        ),
        workers=config.workers,
    )

    summary = build_summary(records, config)
    manifest_rows = build_frame_manifest_rows(records)
    with config.output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, ensure_ascii=False, indent=2)
    return {
        "summary": summary,
        "records": manifest_rows,
    }


def discover_video_jobs(sources: Sequence[VideoSource]) -> List[Tuple[str, Path, Path]]:
    jobs: List[Tuple[str, Path, Path]] = []
    for source in sources:
        if not source.root.exists():
            raise FileNotFoundError(f"source video directory not found: {source.root}")
        files = sorted(
            path
            for path in source.root.rglob("*")
            if path.is_file() and path.suffix.lower() in COMMON_VIDEO_EXTENSIONS
        )
        jobs.extend((source.name, source.root, path) for path in files)
    return jobs


def process_video_job(
    *,
    dataset_name: str,
    source_root: Path,
    video_path: Path,
    config: VideoFrameExtractionConfig,
) -> Dict[str, Any]:
    relative_video_path = video_path.relative_to(source_root)
    video_key = relative_video_path.with_suffix("")
    output_dir = config.output_images_root / dataset_name / video_key
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_paths = expected_output_paths(output_dir, config.frames_per_video, config.image_extension)
    if not config.overwrite and all(path.exists() for path in existing_paths):
        return build_existing_record(
            dataset_name=dataset_name,
            source_root=source_root,
            video_path=video_path,
            output_dir=output_dir,
            image_paths=existing_paths,
            frames_per_video=config.frames_per_video,
        )

    backend = detect_backend()
    if backend is None:
        raise RuntimeError("No supported video backend found. Install ffmpeg/ffprobe or opencv-python.")

    try:
        if backend == "ffmpeg":
            record = extract_with_ffmpeg(
                dataset_name=dataset_name,
                source_root=source_root,
                video_path=video_path,
                output_dir=output_dir,
                frames_per_video=config.frames_per_video,
                image_extension=config.image_extension,
            )
        else:
            record = extract_with_opencv(
                dataset_name=dataset_name,
                source_root=source_root,
                video_path=video_path,
                output_dir=output_dir,
                frames_per_video=config.frames_per_video,
                image_extension=config.image_extension,
            )
        record["backend"] = backend
        return record
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "source_video_path": str(video_path),
            "relative_video_path": str(relative_video_path),
            "status": "error",
            "error": str(exc),
            "frames_per_video_requested": config.frames_per_video,
            "frames": [],
        }


def extract_with_ffmpeg(
    *,
    dataset_name: str,
    source_root: Path,
    video_path: Path,
    output_dir: Path,
    frames_per_video: int,
    image_extension: str,
) -> Dict[str, Any]:
    duration_sec = probe_duration_ffmpeg(video_path)
    timestamps = compute_uniform_timestamps(duration_sec, frames_per_video)
    frame_records = []
    for index, timestamp_sec in enumerate(timestamps):
        output_path = output_dir / f"frame_{index:02d}{image_extension}"
        extract_one_frame_ffmpeg(video_path, output_path, timestamp_sec)
        frame_records.append(
            {
                "frame_index": index,
                "timestamp_sec": round(timestamp_sec, 6),
                "image_path": str(output_path),
                "relative_image_path": str(output_path.relative_to(output_dir.parent.parent)),
            }
        )
    return build_video_record(
        dataset_name=dataset_name,
        source_root=source_root,
        video_path=video_path,
        output_dir=output_dir,
        duration_sec=duration_sec,
        status="ok",
        frame_records=frame_records,
    )


def extract_with_opencv(
    *,
    dataset_name: str,
    source_root: Path,
    video_path: Path,
    output_dir: Path,
    frames_per_video: int,
    image_extension: str,
) -> Dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
        frame_indices = compute_uniform_frame_indices(total_frames, frames_per_video)

        frame_records = []
        for index, frame_index in enumerate(frame_indices):
            output_path = output_dir / f"frame_{index:02d}{image_extension}"
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {frame_index} from {video_path}")
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"failed to write frame image: {output_path}")
            timestamp_sec = frame_index / fps if fps > 0 else 0.0
            frame_records.append(
                {
                    "frame_index": index,
                    "source_frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 6),
                    "image_path": str(output_path),
                    "relative_image_path": str(output_path.relative_to(output_dir.parent.parent)),
                }
            )
        return build_video_record(
            dataset_name=dataset_name,
            source_root=source_root,
            video_path=video_path,
            output_dir=output_dir,
            duration_sec=duration_sec,
            status="ok",
            frame_records=frame_records,
        )
    finally:
        cap.release()


def build_video_record(
    *,
    dataset_name: str,
    source_root: Path,
    video_path: Path,
    output_dir: Path,
    duration_sec: float,
    status: str,
    frame_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    relative_video_path = video_path.relative_to(source_root)
    return {
        "dataset": dataset_name,
        "source_video_path": str(video_path),
        "relative_video_path": str(relative_video_path),
        "output_dir": str(output_dir),
        "duration_sec": round(duration_sec, 6),
        "status": status,
        "frame_count_extracted": len(frame_records),
        "frames": frame_records,
    }


def build_existing_record(
    *,
    dataset_name: str,
    source_root: Path,
    video_path: Path,
    output_dir: Path,
    image_paths: List[Path],
    frames_per_video: int,
) -> Dict[str, Any]:
    relative_video_path = video_path.relative_to(source_root)
    return {
        "dataset": dataset_name,
        "source_video_path": str(video_path),
        "relative_video_path": str(relative_video_path),
        "output_dir": str(output_dir),
        "duration_sec": None,
        "status": "reused_existing",
        "frame_count_extracted": len(image_paths),
        "frames": [
            {
                "frame_index": index,
                "timestamp_sec": None,
                "image_path": str(path),
                "relative_image_path": str(path.relative_to(output_dir.parent.parent)),
            }
            for index, path in enumerate(image_paths[:frames_per_video])
        ],
    }


def build_summary(records: Sequence[Dict[str, Any]], config: VideoFrameExtractionConfig) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    dataset_counts: Dict[str, int] = {}
    extracted_image_count = 0

    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        dataset = str(record.get("dataset", "unknown"))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        extracted_image_count += int(record.get("frame_count_extracted", 0))

    return {
        "video_count": len(records),
        "frames_per_video": config.frames_per_video,
        "total_extracted_images": extracted_image_count,
        "workers": config.workers,
        "overwrite": config.overwrite,
        "output_images_root": str(config.output_images_root),
        "output_manifest_path": str(config.output_manifest_path),
        "status_counts": status_counts,
        "dataset_counts": dataset_counts,
    }


def build_frame_manifest_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for record in records:
        source = str(record.get("source_video_path", ""))
        for frame in record.get("frames", []):
            image_path = str(frame.get("image_path", ""))
            if not image_path:
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "source": source,
                }
            )
    return rows


def expected_output_paths(output_dir: Path, frames_per_video: int, image_extension: str) -> List[Path]:
    return [output_dir / f"frame_{index:02d}{image_extension}" for index in range(frames_per_video)]


def compute_uniform_timestamps(duration_sec: float, frames_per_video: int) -> List[float]:
    if frames_per_video <= 0:
        return []
    if duration_sec <= 0:
        return [0.0 for _ in range(frames_per_video)]
    epsilon = min(1e-3, duration_sec / max(frames_per_video * 1000, 1))
    return [min(duration_sec - epsilon, duration_sec * ((index + 0.5) / frames_per_video)) for index in range(frames_per_video)]


def compute_uniform_frame_indices(total_frames: int, frames_per_video: int) -> List[int]:
    if frames_per_video <= 0:
        return []
    if total_frames <= 1:
        return [0 for _ in range(frames_per_video)]
    indices: List[int] = []
    for index in range(frames_per_video):
        raw = total_frames * ((index + 0.5) / frames_per_video) - 0.5
        indices.append(max(0, min(total_frames - 1, int(round(raw)))))
    return indices


def detect_backend() -> Optional[str]:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return "ffmpeg"
    try:
        import cv2  # noqa: F401

        return "opencv"
    except Exception:
        return None


def probe_duration_ffmpeg(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        return 0.0
    return float(output)


def extract_one_frame_ffmpeg(video_path: Path, output_path: Path, timestamp_sec: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp_sec:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parallel_map(items: Sequence[Any], fn: Any, workers: int) -> List[Any]:
    if not items:
        return []
    worker_count = max(1, min(workers, len(items)))
    if worker_count == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fn, items))
