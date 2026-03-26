from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_CAPTION_JSONL,
    DEFAULT_GROUNDING_JSON,
    DEFAULT_RESULTS_DIR,
    ERP_MULTIMODAL_SYSTEM_PROMPT,
)
from .normalizers import normalize_caption_records, normalize_grounding_records, normalize_results_final_v2

DEFAULT_WORKERS = min(32, (os.cpu_count() or 4) + 4)
TRAINING_FORMAT_SIMPLE = "simple"
TRAINING_FORMAT_MULTIMODAL_BLOCKS = "multimodal_blocks"
TRAINING_FORMATS = {TRAINING_FORMAT_SIMPLE, TRAINING_FORMAT_MULTIMODAL_BLOCKS}


@dataclass
class BuildConfig:
    results_dir: Path = Path(DEFAULT_RESULTS_DIR)
    caption_jsonl: Path = Path(DEFAULT_CAPTION_JSONL)
    grounding_json: Path = Path(DEFAULT_GROUNDING_JSON)
    output_dir: Path = Path("outputs/sft_merge")
    include_full_caption: bool = True
    drop_missing_images: bool = False
    workers: int = DEFAULT_WORKERS
    training_format: str = TRAINING_FORMAT_SIMPLE
    max_results_records: Optional[int] = None
    max_caption_records: Optional[int] = None
    max_grounding_records: Optional[int] = None


def build_dataset(config: BuildConfig) -> Dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_items = normalize_results_final_v2(
        config.results_dir,
        drop_missing_images=config.drop_missing_images,
        workers=config.workers,
    )
    caption_items = normalize_caption_records(
        config.caption_jsonl,
        include_full_caption=config.include_full_caption,
        drop_missing_images=config.drop_missing_images,
        workers=config.workers,
    )
    grounding_items = normalize_grounding_records(
        config.grounding_json,
        drop_missing_images=config.drop_missing_images,
        workers=config.workers,
    )

    if config.max_results_records is not None:
        results_items = results_items[: config.max_results_records]
    if config.max_caption_records is not None:
        caption_items = caption_items[: config.max_caption_records]
    if config.max_grounding_records is not None:
        grounding_items = grounding_items[: config.max_grounding_records]

    if config.training_format not in TRAINING_FORMATS:
        raise ValueError(f"unsupported training_format: {config.training_format}")

    merged_items = results_items + caption_items + grounding_items
    training_items = [project_training_item(item, config.training_format) for item in merged_items]
    training_filename = build_training_filename(config.training_format)

    _write_jsonl(output_dir / "normalized_results_final_v2.jsonl", results_items)
    _write_jsonl(output_dir / "normalized_caption_sft.jsonl", caption_items)
    _write_jsonl(output_dir / "normalized_grounding_sft.jsonl", grounding_items)
    _write_jsonl(output_dir / "merged_sft.jsonl", merged_items)
    _write_json(output_dir / training_filename, training_items)

    stats = {
        "results_final_v2_count": len(results_items),
        "caption_sft_count": len(caption_items),
        "grounding_count": len(grounding_items),
        "merged_total_count": len(merged_items),
        "training_data_count": len(training_items),
        "training_format": config.training_format,
        "include_full_caption": config.include_full_caption,
        "drop_missing_images": config.drop_missing_images,
        "workers": config.workers,
        "input_paths": {
            "results_dir": str(config.results_dir),
            "caption_jsonl": str(config.caption_jsonl),
            "grounding_json": str(config.grounding_json),
        },
        "output_files": {
            "results": str(output_dir / "normalized_results_final_v2.jsonl"),
            "caption": str(output_dir / "normalized_caption_sft.jsonl"),
            "grounding": str(output_dir / "normalized_grounding_sft.jsonl"),
            "merged": str(output_dir / "merged_sft.jsonl"),
            "training_data": str(output_dir / training_filename),
        },
    }
    _write_json(output_dir / "stats.json", stats)
    return stats


def project_training_item(item: Dict[str, object], training_format: str) -> Dict[str, object]:
    if training_format == TRAINING_FORMAT_MULTIMODAL_BLOCKS:
        return project_multimodal_training_item(item)
    return project_simple_training_item(item)


def project_simple_training_item(item: Dict[str, object]) -> Dict[str, object]:
    messages = []
    image_count = len(item.get("images", [])) if isinstance(item.get("images"), list) else 0
    user_image_tag_added = False

    for raw_message in item.get("messages", []):
        if not isinstance(raw_message, dict):
            continue
        role = raw_message.get("role")
        content = str(raw_message.get("content", ""))
        if role == "user" and image_count > 0 and not user_image_tag_added:
            content = ensure_image_tag(content)
            user_image_tag_added = True
        messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "images": list(item.get("images", [])) if isinstance(item.get("images"), list) else [],
    }


def project_multimodal_training_item(item: Dict[str, object]) -> Dict[str, object]:
    images = list(item.get("images", [])) if isinstance(item.get("images"), list) else []
    projected_messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": ERP_MULTIMODAL_SYSTEM_PROMPT}],
        }
    ]

    for raw_message in item.get("messages", []):
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role", "")).strip()
        content = str(raw_message.get("content", ""))
        if not role:
            continue
        if role == "user":
            blocks: List[Dict[str, str]] = []
            text_content = strip_leading_image_tag(content)
            if text_content:
                blocks.append({"type": "text", "text": text_content})
            for _ in images:
                blocks.append({"type": "image"})
            if not blocks:
                continue
            projected_messages.append({"role": "user", "content": blocks})
            continue

        projected_messages.append(
            {
                "role": role,
                "content": [{"type": "text", "text": content}],
            }
        )

    return {
        "id": str(item.get("id", "")),
        "messages": projected_messages,
        "images": images,
    }


def ensure_image_tag(content: str) -> str:
    stripped = content.lstrip()
    if stripped.startswith("<image>"):
        return content
    return f"<image>{content}"


def strip_leading_image_tag(content: str) -> str:
    stripped = content.lstrip()
    if stripped.startswith("<image>"):
        return stripped[len("<image>") :].lstrip()
    return content


def build_training_filename(training_format: str) -> str:
    if training_format == TRAINING_FORMAT_MULTIMODAL_BLOCKS:
        return "training_data_multimodal_blocks.json"
    return "training_data.json"


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
