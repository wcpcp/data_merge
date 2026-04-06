from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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

_WORKER_IMAGE_ROOT: Optional[Path] = None
_WORKER_DATASET_ROOT: Optional[Path] = None
_WORKER_DATASET_NAME: str = ""
_WORKER_METADATA_INDEX: Dict[str, Dict[str, Any]] = {}
_WORKER_RESIZE_WIDTH: int = 2048
_WORKER_RESIZE_HEIGHT: int = 1024


@dataclass
class ImageManifestConfig:
    dataset_root: Path
    output_manifest_path: Path
    dataset_name: str = "real_360_test"
    workers: int = min(16, (os.cpu_count() or 4))
    resize_width: int = 2048
    resize_height: int = 1024
    progress_every: int = 200


def build_image_manifest(config: ImageManifestConfig) -> Dict[str, Any]:
    dataset_root = config.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    image_root = resolve_image_root(dataset_root)
    metadata_index = load_metadata_index(dataset_root)

    print(
        f"[image_manifest] dataset={config.dataset_name} image_root={image_root} "
        f"images=streaming metadata_rows={len(metadata_index)} workers={config.workers}",
        flush=True,
    )

    records = parallel_build_image_records_streaming(
        discover_image_paths(image_root),
        image_root=image_root,
        dataset_root=dataset_root,
        dataset_name=config.dataset_name,
        metadata_index=metadata_index,
        resize_width=config.resize_width,
        resize_height=config.resize_height,
        workers=config.workers,
        progress_every=config.progress_every,
        progress_label=f"image_manifest:{config.dataset_name}",
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


def discover_image_paths(image_root: Path) -> Iterator[Path]:
    for root, dirnames, filenames in os.walk(image_root):
        dirnames.sort()
        root_path = Path(root)
        for filename in sorted(filenames):
            path = root_path / filename
            if path.suffix.lower() in COMMON_IMAGE_EXTENSIONS and path.is_file():
                yield path


def load_metadata_index(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    metadata_paths = [
        dataset_root / "search_results.jsonl",
        dataset_root / "metadata.jsonl",
    ]
    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for key in metadata_keys(row):
                    index.setdefault(key, row)
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
        record["id"] = matched.get("id")
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
        record["asset_url"] = matched.get("asset_url")
        record["collection"] = matched.get("collection")
        record["datetime"] = matched.get("datetime")
        record["asset_key"] = matched.get("asset_key")
        record["erp_reason"] = matched.get("erp_reason")
        record["quality_grade"] = matched.get("quality_grade")
        record["quality_score_value"] = matched.get("quality_score_value")
        record["lon"] = matched.get("lon")
        record["lat"] = matched.get("lat")
        record["image_name"] = matched.get("image_name")
        record["local_path"] = matched.get("local_path")
        record["source_metadata_csv"] = matched.get("source_metadata_csv")
        record["source_region_dir"] = matched.get("source_region_dir")
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
    for key in ["asset_url", "landing_url", "file_url", "source_id", "id", "source", "relative_image_path", "image_path"]:
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


def metadata_keys(row: Dict[str, object]) -> List[str]:
    keys: List[str] = []

    def add(value: object) -> None:
        if not isinstance(value, str):
            return
        stem = Path(value.strip()).stem
        if stem and stem not in keys:
            keys.append(stem)

    add(str(row.get("image_name") or ""))
    add(str(row.get("local_path") or ""))
    add(str(row.get("source_id") or ""))
    add(str(row.get("id") or ""))

    stable = stable_stem(row)
    if stable not in keys:
        keys.append(stable)
    return keys


def init_image_record_worker(
    image_root: str,
    dataset_root: str,
    dataset_name: str,
    metadata_index: Dict[str, Dict[str, Any]],
    resize_width: int,
    resize_height: int,
) -> None:
    global _WORKER_IMAGE_ROOT
    global _WORKER_DATASET_ROOT
    global _WORKER_DATASET_NAME
    global _WORKER_METADATA_INDEX
    global _WORKER_RESIZE_WIDTH
    global _WORKER_RESIZE_HEIGHT

    _WORKER_IMAGE_ROOT = Path(image_root)
    _WORKER_DATASET_ROOT = Path(dataset_root)
    _WORKER_DATASET_NAME = dataset_name
    _WORKER_METADATA_INDEX = metadata_index
    _WORKER_RESIZE_WIDTH = resize_width
    _WORKER_RESIZE_HEIGHT = resize_height


def build_image_record_worker(path_str: str) -> Dict[str, Any]:
    if _WORKER_IMAGE_ROOT is None or _WORKER_DATASET_ROOT is None:
        raise RuntimeError("image manifest worker not initialized")
    return build_image_record(
        path=Path(path_str),
        image_root=_WORKER_IMAGE_ROOT,
        dataset_root=_WORKER_DATASET_ROOT,
        dataset_name=_WORKER_DATASET_NAME,
        metadata_index=_WORKER_METADATA_INDEX,
        resize_width=_WORKER_RESIZE_WIDTH,
        resize_height=_WORKER_RESIZE_HEIGHT,
    )


def parallel_map(items: Sequence[Any], fn: Any, workers: int) -> List[Any]:
    if not items:
        return []
    worker_count = max(1, min(workers, len(items)))
    if worker_count == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fn, items))


def parallel_build_image_records_streaming(
    items: Iterable[Path],
    *,
    image_root: Path,
    dataset_root: Path,
    dataset_name: str,
    metadata_index: Dict[str, Dict[str, Any]],
    resize_width: int,
    resize_height: int,
    workers: int,
    progress_every: int,
    progress_label: str,
) -> List[Any]:
    worker_count = max(1, workers)
    progress_step = max(1, progress_every)
    started_at = time.time()

    if worker_count == 1:
        records: List[Any] = []
        for index, item in enumerate(items, start=1):
            records.append(
                build_image_record(
                    path=item,
                    image_root=image_root,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    metadata_index=metadata_index,
                    resize_width=resize_width,
                    resize_height=resize_height,
                )
            )
            if index % progress_step == 0:
                elapsed = time.time() - started_at
                rate = index / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{progress_label}] completed={index} discovered={index} "
                    f"({rate:.2f} img/s, elapsed={elapsed:.1f}s)",
                    flush=True,
                )
        if records:
            elapsed = time.time() - started_at
            rate = len(records) / elapsed if elapsed > 0 else 0.0
            print(
                f"[{progress_label}] completed={len(records)} discovered={len(records)} "
                f"({rate:.2f} img/s, elapsed={elapsed:.1f}s)",
                flush=True,
            )
        return records

    items_iter = iter(items)
    max_pending = max(worker_count * 4, worker_count)
    next_index = 0
    discovered = 0
    completed = 0
    results_by_index: Dict[int, Any] = {}

    def submit_next(executor: ProcessPoolExecutor, pending: Dict[Any, int]) -> bool:
        nonlocal next_index
        nonlocal discovered
        try:
            item = next(items_iter)
        except StopIteration:
            return False
        future = executor.submit(build_image_record_worker, str(item))
        pending[future] = next_index
        next_index += 1
        discovered += 1
        return True

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_image_record_worker,
        initargs=(
            str(image_root),
            str(dataset_root),
            dataset_name,
            metadata_index,
            resize_width,
            resize_height,
        ),
    ) as executor:
        pending: Dict[Any, int] = {}
        while len(pending) < max_pending and submit_next(executor, pending):
            pass

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                results_by_index[index] = future.result()
                completed += 1
                while len(pending) < max_pending and submit_next(executor, pending):
                    pass
            if completed % progress_step == 0 or (not pending and completed == discovered):
                elapsed = time.time() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{progress_label}] completed={completed} discovered={discovered} "
                    f"inflight={len(pending)} ({rate:.2f} img/s, elapsed={elapsed:.1f}s)",
                    flush=True,
                )
    return [results_by_index[index] for index in sorted(results_by_index)]
