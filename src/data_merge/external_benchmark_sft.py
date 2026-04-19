from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlopen

from .config import ERP_MULTIMODAL_SYSTEM_PROMPT


OSR_BENCH_QA_URL = "https://huggingface.co/datasets/UUUserna/OSR-Bench/resolve/main/qa.csv?download=true"
OSR_BENCH_IMAGE_BASE_URL = "https://huggingface.co/datasets/UUUserna/OSR-Bench/resolve/main"

THINKING_IN_360_SOURCES = (
    {
        "source_name": "hos_sft_panorama",
        "display_name": "Thinking in 360 / HOS-SFT Panorama",
        "zip_url": "https://huggingface.co/datasets/humanoid-vstar/hos_sft_panorama/resolve/main/hos_sft_panorama.zip?download=true",
        "zip_filename": "hos_sft_panorama.zip",
    },
    {
        "source_name": "hps_sft_panorama",
        "display_name": "Thinking in 360 / HPS-SFT Panorama",
        "zip_url": "https://huggingface.co/datasets/humanoid-vstar/hps_sft_panorama/resolve/main/hps_sft_pano.zip?download=true",
        "zip_filename": "hps_sft_pano.zip",
    },
)

PANOENV_DATASET_CANDIDATES = ("7zkk/PanoEnv", "guangmulizi/PanoEnv")

OSR_BENCH_NOTES = {
    "official_training_data_available": False,
    "recommended_for_panorama_training": False,
    "conversion_supported_with_opt_in": True,
    "benchmark_safe_for_same_benchmark": False,
    "summary": (
        "OSR-Bench is primarily a benchmark release. The public package exposes panorama images plus a single "
        "qa.csv file, but no official QA train/val/test split for model fitting. You can still force-convert it "
        "into training format, but then OSR-Bench evaluation is no longer clean."
    ),
    "official_sources": [
        "https://huggingface.co/datasets/UUUserna/OSR-Bench",
        OSR_BENCH_QA_URL,
        "https://arxiv.org/abs/2505.11907",
    ],
}

THINKING_IN_360_NOTES = {
    "official_training_data_available": True,
    "recommended_for_panorama_training": True,
    "conversion_supported_with_opt_in": True,
    "benchmark_safe_for_same_benchmark": True,
    "summary": (
        "Use the official panorama SFT releases for training. They align with panorama-native modeling better than "
        "the older perspective-view HOS/HPS SFT dumps. There is no need to fall back to the RL trajectories unless "
        "the panorama SFT package becomes unavailable."
    ),
    "official_sources": [
        "https://humanoid-vstar.github.io/",
        "https://huggingface.co/datasets/humanoid-vstar/hos_sft_panorama",
        "https://huggingface.co/datasets/humanoid-vstar/hps_sft_panorama",
        "https://huggingface.co/datasets/humanoid-vstar/hos-sft",
        "https://huggingface.co/datasets/humanoid-vstar/hps-sft",
    ],
}

PANOENV_NOTES = {
    "official_training_data_available": True,
    "recommended_for_panorama_training": True,
    "conversion_supported_with_opt_in": True,
    "benchmark_safe_for_same_benchmark": True,
    "summary": (
        "Use only the official train split from PanoEnv for training. Validation and test should stay untouched "
        "for benchmark reporting."
    ),
    "official_sources": [
        "https://huggingface.co/datasets/guangmulizi/PanoEnv",
        "https://huggingface.co/datasets/7zkk/PanoEnv",
    ],
}


@dataclass
class ExternalBenchmarkBuildConfig:
    output_dir: Path
    cache_dir: Path
    max_osr_records: Optional[int] = None
    max_thinking_records: Optional[int] = None
    max_panoenv_records: Optional[int] = None
    include_osr_bench: bool = False
    include_thinking_in_360: bool = True
    include_panoenv: bool = True
    panoenv_allowed_question_types: Tuple[str, ...] = ("open_ended",)


def build_external_benchmark_training_sets(config: ExternalBenchmarkBuildConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "cache_dir": str(config.cache_dir),
        "files": {},
        "counts": {},
        "notes": {
            "osr_bench": OSR_BENCH_NOTES,
            "thinking_in_360": THINKING_IN_360_NOTES,
            "panoenv": PANOENV_NOTES,
        },
    }

    osr_train_path = config.output_dir / "osr_bench_train_multimodal_blocks.json"
    osr_val_path = config.output_dir / "osr_bench_validation_multimodal_blocks.json"
    osr_test_path = config.output_dir / "osr_bench_test_multimodal_blocks.json"
    if config.include_osr_bench:
        osr_rows = build_osr_bench_training_rows(config.cache_dir, config.max_osr_records)
        osr_split = split_grouped_training_rows(osr_rows)
        stats["counts"]["osr_bench"] = {
            "train": len(osr_split["train"]),
            "validation": len(osr_split["validation"]),
            "test": len(osr_split["test"]),
            "total": sum(len(rows) for rows in osr_split.values()),
            "unique_images": len({group_id for group_id, _ in osr_rows}),
        }
        _write_json(osr_train_path, osr_split["train"])
        _write_json(osr_val_path, osr_split["validation"])
        _write_json(osr_test_path, osr_split["test"])
    else:
        stats["counts"]["osr_bench"] = {"train": 0, "validation": 0, "test": 0, "total": 0, "unique_images": 0}
        stats["notes"]["osr_bench"]["export_status"] = "skipped_by_default_because_no_official_training_split"
        _write_json(osr_train_path, [])
        _write_json(osr_val_path, [])
        _write_json(osr_test_path, [])
    stats["files"]["osr_bench"] = {
        "train": str(osr_train_path),
        "validation": str(osr_val_path),
        "test": str(osr_test_path),
    }

    thinking_path = config.output_dir / "thinking_in_360_training_multimodal_blocks.json"
    if config.include_thinking_in_360:
        thinking_items = build_thinking_in_360_training_items(config.cache_dir, config.max_thinking_records)
        stats["counts"]["thinking_in_360"] = len(thinking_items)
    else:
        thinking_items = []
        stats["counts"]["thinking_in_360"] = 0
        stats["notes"]["thinking_in_360"]["export_status"] = "skipped_by_request"
    _write_json(thinking_path, thinking_items)
    stats["files"]["thinking_in_360"] = str(thinking_path)

    panoenv_path = config.output_dir / "panoenv_training_multimodal_blocks.json"
    if config.include_panoenv:
        panoenv_items, panoenv_stats = build_panoenv_training_items(
            config.cache_dir,
            config.max_panoenv_records,
            allowed_question_types=config.panoenv_allowed_question_types,
        )
        stats["counts"]["panoenv"] = panoenv_stats
        stats["notes"]["panoenv"]["resolved_dataset"] = panoenv_stats["resolved_dataset"]
        stats["notes"]["panoenv"]["quality_filter"] = {
            "allowed_question_types": list(config.panoenv_allowed_question_types),
        }
    else:
        panoenv_items = []
        stats["counts"]["panoenv"] = 0
        stats["notes"]["panoenv"]["export_status"] = "skipped_by_request"
    _write_json(panoenv_path, panoenv_items)
    stats["files"]["panoenv"] = str(panoenv_path)

    stats_path = config.output_dir / "external_benchmark_stats.json"
    stats["files"]["stats"] = str(stats_path)
    _write_json(stats_path, stats)
    return stats


def build_osr_bench_training_rows(cache_dir: Path, max_records: Optional[int]) -> List[Tuple[str, Dict[str, Any]]]:
    dataset_cache = cache_dir / "osr_bench"
    dataset_cache.mkdir(parents=True, exist_ok=True)
    qa_csv_path = dataset_cache / "qa.csv"
    _download_to_path(OSR_BENCH_QA_URL, qa_csv_path)

    rows: List[Tuple[str, Dict[str, Any]]] = []
    with qa_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if max_records is not None and len(rows) >= max_records:
                break
            image_id = str(row.get("image_id", "")).strip()
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not image_id or not question or not answer:
                continue
            image_path = dataset_cache / image_id
            _download_to_path(f"{OSR_BENCH_IMAGE_BASE_URL}/{image_id}?download=true", image_path)
            group_id = safe_id(Path(image_id).with_suffix("").as_posix())
            item = build_single_turn_training_item(
                record_id=f"osr_bench:{group_id}:{row.get('turn_id', row_index)}",
                question=question,
                answer=answer,
                image_paths=[image_path],
                system_prompt=ERP_MULTIMODAL_SYSTEM_PROMPT,
            )
            rows.append(
                (
                    group_id,
                    item,
                )
            )
    return rows


def build_thinking_in_360_training_items(cache_dir: Path, max_records: Optional[int]) -> List[Dict[str, Any]]:
    dataset_cache = cache_dir / "thinking_in_360"
    dataset_cache.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for source in THINKING_IN_360_SOURCES:
        if max_records is not None and len(items) >= max_records:
            break

        zip_path = dataset_cache / source["zip_filename"]
        extract_dir = dataset_cache / source["source_name"]

        _download_to_path(str(source["zip_url"]), zip_path)
        _extract_zip_once(zip_path, extract_dir)

        manifest_paths = discover_manifest_paths(extract_dir)
        if not manifest_paths:
            raise RuntimeError(f"No JSON/JSONL manifest found after extracting {zip_path}")

        for manifest_path in manifest_paths:
            for row_index, row in enumerate(iter_manifest_rows(manifest_path)):
                if max_records is not None and len(items) >= max_records:
                    break
                row_items = build_thinking_records(
                    row=row,
                    record_index=row_index,
                    source_name=str(source["source_name"]),
                    extract_dir=extract_dir,
                    manifest_path=manifest_path,
                )
                if max_records is not None:
                    remaining = max_records - len(items)
                    row_items = row_items[:remaining]
                items.extend(row_items)
    return items


def build_panoenv_training_items(
    cache_dir: Path,
    max_records: Optional[int],
    allowed_question_types: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset_name, train_split = load_panoenv_train_split()
    dataset_cache = cache_dir / "panoenv"
    image_cache = dataset_cache / "images"
    image_cache.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    image_count = 0
    raw_question_count = 0
    for row in train_split:
        if max_records is not None and len(items) >= max_records:
            break
        image_count += 1
        raw_question_count += len(row.get("questions", [])) if isinstance(row.get("questions"), list) else 0
        image_path = save_panoenv_row_image(row, image_cache)
        row_items = build_panoenv_training_items_from_row(
            row,
            image_path,
            allowed_question_types=set(allowed_question_types),
        )
        if max_records is not None:
            remaining = max_records - len(items)
            row_items = row_items[:remaining]
        items.extend(row_items)

    return items, {
        "filtered_export_count": len(items),
        "scanned_image_count": image_count,
        "raw_question_count_seen": raw_question_count,
        "resolved_dataset": dataset_name,
    }


def build_single_turn_training_item(
    record_id: str,
    question: str,
    answer: str,
    image_paths: Sequence[Path],
    system_prompt: str,
) -> Dict[str, Any]:
    return {
        "id": record_id,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": question}] + [{"type": "image"} for _ in image_paths],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        "images": [str(path) for path in image_paths],
    }


def build_thinking_records(
    row: Dict[str, Any],
    record_index: int,
    source_name: str,
    extract_dir: Path,
    manifest_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    outputs = row.get("outputs")
    if isinstance(outputs, list):
        return build_thinking_records_from_task_outputs(
            row=row,
            record_index=record_index,
            source_name=source_name,
            extract_dir=extract_dir,
            manifest_path=manifest_path,
        )

    item = build_thinking_record(
        row=row,
        record_index=record_index,
        source_name=source_name,
        extract_dir=extract_dir,
    )
    return [item] if item is not None else []


def build_thinking_record(
    row: Dict[str, Any],
    record_index: int,
    source_name: str,
    extract_dir: Path,
) -> Optional[Dict[str, Any]]:
    existing_messages = row.get("messages")
    if isinstance(existing_messages, list) and existing_messages:
        return build_thinking_record_from_messages(
            row=row,
            record_index=record_index,
            source_name=source_name,
            extract_dir=extract_dir,
        )

    conversations = row.get("conversations", [])
    if not isinstance(conversations, list) or not conversations:
        return None

    resolved_images = resolve_row_images(row, extract_dir)
    if not resolved_images:
        return None

    system_prompt = str(row.get("system", "")).strip() or ERP_MULTIMODAL_SYSTEM_PROMPT
    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    first_user_message = True
    for conversation in conversations:
        if not isinstance(conversation, dict):
            continue
        role = normalize_sharegpt_role(conversation.get("from"))
        value = str(conversation.get("value", ""))
        if not role:
            continue
        if role == "user":
            blocks = sharegpt_user_text_to_blocks(value, len(resolved_images), first_user_message=first_user_message)
            first_user_message = False
            if not blocks:
                continue
            messages.append({"role": "user", "content": blocks})
            continue
        messages.append({"role": role, "content": [{"type": "text", "text": value}]})

    return {
        "id": f"thinking_in_360:{source_name}:{record_index:06d}",
        "messages": messages,
        "images": [str(path) for path in resolved_images],
    }


def build_thinking_records_from_task_outputs(
    row: Dict[str, Any],
    record_index: int,
    source_name: str,
    extract_dir: Path,
    manifest_path: Optional[Path],
) -> List[Dict[str, Any]]:
    image_paths = resolve_row_images(row, extract_dir, manifest_path=manifest_path)
    if not image_paths:
        return []
    task = str(row.get("task", "")).strip()
    outputs = row.get("outputs", [])
    if not isinstance(outputs, list):
        return []

    items: List[Dict[str, Any]] = []
    for output_index, output in enumerate(outputs):
        if not isinstance(output, dict):
            continue
        input_angles = output.get("input_angles", [])
        yaw, pitch = normalize_input_angles(input_angles)
        assistant_text = str(output.get("content", "")).strip() or str(output.get("action", "")).strip()
        if not assistant_text:
            continue
        user_text = build_thinking_task_prompt(task, yaw, pitch)
        items.append(
            build_single_turn_training_item(
                record_id=f"thinking_in_360:{source_name}:{record_index:06d}:step{output_index:03d}",
                question=user_text,
                answer=assistant_text,
                image_paths=image_paths,
                system_prompt=ERP_MULTIMODAL_SYSTEM_PROMPT,
            )
        )
    return items


def build_thinking_record_from_messages(
    row: Dict[str, Any],
    record_index: int,
    source_name: str,
    extract_dir: Path,
) -> Optional[Dict[str, Any]]:
    resolved_images = resolve_row_images(row, extract_dir)
    messages = row.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return None
    projected = []
    has_system = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role == "system":
            has_system = True
        if isinstance(content, list):
            projected.append({"role": role, "content": content})
        else:
            projected.append({"role": role, "content": [{"type": "text", "text": str(content or "")}]})
    if not has_system:
        projected.insert(0, {"role": "system", "content": [{"type": "text", "text": ERP_MULTIMODAL_SYSTEM_PROMPT}]})
    return {
        "id": str(row.get("id", f"thinking_in_360:{source_name}:{record_index:06d}")),
        "messages": projected,
        "images": [str(path) for path in resolved_images],
    }


def build_panoenv_training_items_from_row(
    row: Dict[str, Any],
    image_path: Path,
    allowed_question_types: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    env_name = str(row.get("env", "")).strip()
    image_id = str(row.get("image_id", "")).strip()
    questions = row.get("questions", [])
    if not env_name or not image_id or not isinstance(questions, list):
        return []

    items: List[Dict[str, Any]] = []
    for question_entry in questions:
        if not isinstance(question_entry, dict):
            continue
        question_type = str(question_entry.get("question_type", "")).strip()
        if allowed_question_types is not None and question_type not in allowed_question_types:
            continue
        question_text = str(question_entry.get("question", "")).strip()
        answer_text = stringify_answer(question_entry.get("answer"))
        if not question_text or not answer_text:
            continue
        question_id = question_entry.get("question_id", len(items))
        items.append(
            build_single_turn_training_item(
                record_id=f"panoenv:{safe_id(env_name)}:{safe_id(image_id)}:q{question_id}",
                question=question_text,
                answer=answer_text,
                image_paths=[image_path],
                system_prompt=ERP_MULTIMODAL_SYSTEM_PROMPT,
            )
        )
    return items


def load_panoenv_train_split() -> Tuple[str, Iterable[Dict[str, Any]]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised through runtime error path
        raise RuntimeError(
            "PanoEnv conversion requires the `datasets` package because the official release is loaded through "
            "the dataset loader rather than a simple JSONL file. Install it with `pip install datasets pillow`, "
            "or rerun this script with `--skip-panoenv`."
        ) from exc

    last_exc: Optional[Exception] = None
    for dataset_name in PANOENV_DATASET_CANDIDATES:
        try:
            split = load_dataset(dataset_name, split="train")
            first_row = split[0]
            if "questions" not in first_row or "image" not in first_row:
                raise RuntimeError(
                    f"{dataset_name} resolved, but the loaded train split does not expose the expected "
                    "`image` + `questions` fields."
                )
            return dataset_name, split
        except Exception as exc:  # pragma: no cover - exercised against remote datasets only
            last_exc = exc

    raise RuntimeError(
        "Unable to load an official PanoEnv train split from the known dataset names: "
        + ", ".join(PANOENV_DATASET_CANDIDATES)
    ) from last_exc


def save_panoenv_row_image(row: Dict[str, Any], image_cache: Path) -> Path:
    env_name = safe_id(str(row.get("env", "")))
    image_id = safe_id(str(row.get("image_id", "")))
    target_path = image_cache / env_name / f"{image_id}.jpg"
    if target_path.exists():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    image_obj = row.get("image")
    if image_obj is None:
        raise RuntimeError(f"PanoEnv row is missing image data for {env_name}/{image_id}")

    if hasattr(image_obj, "save"):
        converted = image_obj
        if hasattr(image_obj, "mode") and getattr(image_obj, "mode", "RGB") not in {"RGB", "L"}:
            converted = image_obj.convert("RGB")
        converted.save(target_path)
        return target_path

    if isinstance(image_obj, dict):
        path_value = image_obj.get("path")
        if isinstance(path_value, str) and path_value:
            source_path = Path(path_value)
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                return target_path
        bytes_value = image_obj.get("bytes")
        if isinstance(bytes_value, (bytes, bytearray)):
            target_path.write_bytes(bytes(bytes_value))
            return target_path
        src_value = image_obj.get("src")
        if isinstance(src_value, str) and src_value:
            _download_to_path(src_value, target_path)
            return target_path

    raise RuntimeError(f"Unsupported PanoEnv image payload for {env_name}/{image_id}: {type(image_obj)!r}")


def normalize_sharegpt_role(raw_role: Any) -> str:
    role = str(raw_role or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    return ""


def sharegpt_user_text_to_blocks(text: str, image_count: int, first_user_message: bool) -> List[Dict[str, str]]:
    if image_count <= 0:
        cleaned = text.replace("<image>", "").strip()
        return [{"type": "text", "text": cleaned}] if cleaned else []

    parts = text.split("<image>")
    blocks: List[Dict[str, str]] = []
    images_used = 0
    if len(parts) > 1:
        for index, part in enumerate(parts):
            if part:
                blocks.append({"type": "text", "text": part})
            if index < len(parts) - 1 and images_used < image_count:
                blocks.append({"type": "image"})
                images_used += 1
    else:
        stripped = text.strip()
        if stripped:
            blocks.append({"type": "text", "text": stripped})

    if first_user_message and images_used < image_count:
        while images_used < image_count:
            blocks.append({"type": "image"})
            images_used += 1
    return blocks


def resolve_row_images(row: Dict[str, Any], extract_dir: Path, manifest_path: Optional[Path] = None) -> List[Path]:
    raw_images = row.get("images")
    if isinstance(raw_images, list):
        image_values = [str(item) for item in raw_images if str(item).strip()]
    else:
        image_value = str(row.get("image", "")).strip()
        image_values = [image_value] if image_value else []

    if not image_values and manifest_path is not None:
        sibling_images = sorted(
            list(manifest_path.parent.glob("*.jpg"))
            + list(manifest_path.parent.glob("*.jpeg"))
            + list(manifest_path.parent.glob("*.png"))
            + list(manifest_path.parent.glob("*.webp"))
        )
        return sibling_images

    resolved_images: List[Path] = []
    for raw_image in image_values:
        resolved = resolve_extracted_image_path(extract_dir, raw_image)
        if resolved is None:
            raise FileNotFoundError(f"missing extracted image for Thinking in 360 sample: {raw_image}")
        resolved_images.append(resolved)
    return resolved_images


def resolve_extracted_image_path(extract_dir: Path, relative_path: str) -> Optional[Path]:
    relative = Path(relative_path)
    direct = extract_dir / relative
    if direct.exists():
        return direct
    nested = extract_dir / relative.name
    if nested.exists():
        return nested
    candidates = list(extract_dir.rglob(relative.name))
    if candidates:
        return candidates[0]
    return None


def discover_manifest_paths(extract_dir: Path) -> List[Path]:
    manifests = sorted(path for path in extract_dir.rglob("*.jsonl") if path.is_file())
    if manifests:
        return manifests
    return sorted(path for path in extract_dir.rglob("*.json") if path.is_file() and path.name != ".complete")


def iter_manifest_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
        return

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        for key in ("data", "items", "records", "rows", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row
                return


def split_grouped_training_rows(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    split_rows = {"train": [], "validation": [], "test": []}
    for group_id, item in rows:
        split_name = assign_group_to_split(group_id)
        split_rows[split_name].append(item)
    return split_rows


def assign_group_to_split(group_id: str) -> str:
    value = int(hashlib.md5(group_id.encode("utf-8")).hexdigest()[:8], 16) % 11
    if value < 9:
        return "train"
    if value == 9:
        return "validation"
    return "test"


def normalize_input_angles(value: Any) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return 0.0, 0.0
    return 0.0, 0.0


def build_thinking_task_prompt(task: str, yaw: float, pitch: float) -> str:
    task_text = task.strip() if task.strip() else "Complete the visual search task."
    return (
        f"Task: {task_text}\n"
        f"The image is a full ERP panorama. The current virtual view center is at yaw={yaw:g} degrees and pitch={pitch:g} degrees.\n"
        "Decide the next action for this task."
    )


def stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def safe_id(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_", ".", ":"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "unknown"


def _download_to_path(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def _extract_zip_once(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".complete"
    if marker.exists():
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
