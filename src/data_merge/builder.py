from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import DEFAULT_CAPTION_JSONL, DEFAULT_GROUNDING_JSON, DEFAULT_RESULTS_DIR
from .normalizers import normalize_caption_records, normalize_grounding_records, normalize_results_final_v2


@dataclass
class BuildConfig:
    results_dir: Path = Path(DEFAULT_RESULTS_DIR)
    caption_jsonl: Path = Path(DEFAULT_CAPTION_JSONL)
    grounding_json: Path = Path(DEFAULT_GROUNDING_JSON)
    output_dir: Path = Path("outputs/sft_merge")
    include_full_caption: bool = True
    drop_missing_images: bool = False
    max_results_records: Optional[int] = None
    max_caption_records: Optional[int] = None
    max_grounding_records: Optional[int] = None


def build_dataset(config: BuildConfig) -> Dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_items = normalize_results_final_v2(config.results_dir, drop_missing_images=config.drop_missing_images)
    caption_items = normalize_caption_records(
        config.caption_jsonl,
        include_full_caption=config.include_full_caption,
        drop_missing_images=config.drop_missing_images,
    )
    grounding_items = normalize_grounding_records(
        config.grounding_json,
        drop_missing_images=config.drop_missing_images,
    )

    if config.max_results_records is not None:
        results_items = results_items[: config.max_results_records]
    if config.max_caption_records is not None:
        caption_items = caption_items[: config.max_caption_records]
    if config.max_grounding_records is not None:
        grounding_items = grounding_items[: config.max_grounding_records]

    merged_items = results_items + caption_items + grounding_items

    _write_jsonl(output_dir / "normalized_results_final_v2.jsonl", results_items)
    _write_jsonl(output_dir / "normalized_caption_sft.jsonl", caption_items)
    _write_jsonl(output_dir / "normalized_grounding_sft.jsonl", grounding_items)
    _write_jsonl(output_dir / "merged_sft.jsonl", merged_items)

    stats = {
        "results_final_v2_count": len(results_items),
        "caption_sft_count": len(caption_items),
        "grounding_count": len(grounding_items),
        "merged_total_count": len(merged_items),
        "include_full_caption": config.include_full_caption,
        "drop_missing_images": config.drop_missing_images,
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
        },
    }
    _write_json(output_dir / "stats.json", stats)
    return stats


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
