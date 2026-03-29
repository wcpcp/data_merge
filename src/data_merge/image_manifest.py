from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .image_ops import resize_image_in_place


COMMON_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
}

STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ImageManifestConfig:
    dataset_root: Path
    output_manifest_path: Path
    dataset_name: str = "real_360_test"
    workers: int = min(16, (os.cpu_count() or 4))
    resize_width: int = 2048
    resize_height: int = 1024


def build_image_manifest(config: ImageManifestConfig) -> Dict[str, Any]:
    dataset_root = config.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    image_root = resolve_image_root(dataset_root)
    metadata_index = load_search_results_index(dataset_root)
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    )

    records = parallel_map(
        image_paths,
        lambda path: build_image_record(
            path=path,
            image_root=image_root,
            dataset_root=dataset_root,
            dataset_name=config.dataset_name,
            metadata_index=metadata_index,
            resize_width=config.resize_width,
            resize_height=config.resize_height,
        ),
        workers=config.workers,
    )

    summary = build_image_summary(
        dataset_root=dataset_root,
        image_root=image_root,
        dataset_name=config.dataset_name,
        records=records,
        workers=config.workers,
    )
    manifest_rows = [
        {
            "image_path": str(record.get("image_path", "")),
            "source": choose_image_source(record),
            "scene_id": str(record.get("scene_id", "")),
            "viewpoint_id": str(record.get("viewpoint_id", "")),
        }
        for record in records
        if str(record.get("image_path", ""))
    ]
    config.output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, ensure_ascii=False, indent=2)
    return {
        "summary": summary,
        "records": manifest_rows,
    }


def resolve_image_root(dataset_root: Path) -> Path:
    candidate = dataset_root / "images"
    return candidate if candidate.exists() and candidate.is_dir() else dataset_root


def load_search_results_index(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    search_results_path = dataset_root / "search_results.jsonl"
    if not search_results_path.exists():
        return {}

    index: Dict[str, Dict[str, Any]] = {}
    with search_results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[stable_stem(row)] = row
    return index


def build_image_record(
    *,
    path: Path,
    image_root: Path,
    dataset_root: Path,
    dataset_name: str,
    metadata_index: Dict[str, Dict[str, Any]],
    resize_width: int,
    resize_height: int,
) -> Dict[str, Any]:
    resize_image_in_place(path, resize_width, resize_height)
    stat = path.stat()
    relative_image_path = path.relative_to(image_root)
    stem_key = path.stem
    matched = metadata_index.get(stem_key)

    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "image_path": str(path),
        "relative_image_path": str(relative_image_path),
        "source_root": str(dataset_root),
        "image_root": str(image_root),
        "filename": path.name,
        "stem": path.stem,
        "scene_id": path.stem,
        "viewpoint_id": path.stem,
        "suffix": path.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "metadata_match_found": matched is not None,
    }

    if matched is not None:
        record["source"] = matched.get("source")
        record["source_id"] = matched.get("source_id")
        record["provider"] = matched.get("provider")
        record["title"] = matched.get("title")
        record["caption"] = matched.get("caption")
        record["creator"] = matched.get("creator")
        record["license"] = matched.get("license")
        record["license_url"] = matched.get("license_url")
        record["landing_url"] = matched.get("landing_url")
        record["file_url"] = matched.get("file_url")
        record["width"] = matched.get("width")
        record["height"] = matched.get("height")
        record["aspect_ratio"] = matched.get("aspect_ratio")
        record["quality_bucket"] = matched.get("quality_bucket")
        record["is_360"] = matched.get("is_360")
        record["query"] = matched.get("query")
    return record


def build_image_summary(
    *,
    dataset_root: Path,
    image_root: Path,
    dataset_name: str,
    records: Sequence[Dict[str, Any]],
    workers: int,
) -> Dict[str, Any]:
    extension_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    matched_count = 0

    for record in records:
        suffix = str(record.get("suffix", ""))
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        source = str(record.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        if record.get("metadata_match_found"):
            matched_count += 1

    return {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "image_count": len(records),
        "metadata_match_count": matched_count,
        "workers": workers,
        "extension_counts": extension_counts,
        "source_counts": source_counts,
    }


def choose_image_source(record: Dict[str, Any]) -> str:
    for key in ["landing_url", "file_url", "source_id", "source", "relative_image_path", "image_path"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def stable_stem(row: Dict[str, object]) -> str:
    source = str(row.get("source") or "source").strip() or "source"
    source_id = str(row.get("source_id") or row.get("id") or "item").strip() or "item"
    source_id = Path(source_id).stem
    combined = STEM_RE.sub("_", f"{source}__{source_id}").strip("._")
    return combined[:180] or "item"


def parallel_map(items: Sequence[Any], fn: Any, workers: int) -> List[Any]:
    if not items:
        return []
    worker_count = max(1, min(workers, len(items)))
    if worker_count == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fn, items))
