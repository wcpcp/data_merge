from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .caption_parser import parse_caption_sections
from .config import CAPTION_SECTION_PROMPTS, FULL_CAPTION_PROMPT, REALSEE_PUBLIC_ROOT
from .path_utils import remap_path, remap_paths_in_payload


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_results_final_v2(
    results_dir: Path,
    drop_missing_images: bool = False,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not results_dir.exists():
        raise FileNotFoundError(f"results_final_v2 directory not found: {results_dir}")

    canonical_files = sorted(results_dir.rglob("canonical_samples.jsonl"))
    if not canonical_files:
        raise FileNotFoundError(f"no canonical_samples.jsonl found under: {results_dir}")

    for file_items in parallel_map(
        canonical_files,
        lambda file_path: normalize_results_file(
            file_path=file_path,
            results_dir=results_dir,
            drop_missing_images=drop_missing_images,
        ),
        workers=workers,
    ):
        items.extend(file_items)
    return items


def normalize_caption_records(
    caption_jsonl: Path,
    include_full_caption: bool = True,
    drop_missing_images: bool = False,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    if not caption_jsonl.exists():
        raise FileNotFoundError(f"caption jsonl not found: {caption_jsonl}")

    records = load_jsonl(caption_jsonl)
    items: List[Dict[str, Any]] = []

    indexed_records = list(enumerate(records))
    for record_items in parallel_map(
        indexed_records,
        lambda pair: normalize_caption_record(
            record_index=pair[0],
            record=pair[1],
            include_full_caption=include_full_caption,
            drop_missing_images=drop_missing_images,
        ),
        workers=workers,
    ):
        items.extend(record_items)
    return items


def normalize_grounding_records(
    grounding_json: Path,
    drop_missing_images: bool = False,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    if not grounding_json.exists():
        raise FileNotFoundError(f"grounding json not found: {grounding_json}")

    payload = load_json(grounding_json)
    if not isinstance(payload, list):
        raise ValueError("grounding json must contain a list of records")

    items: List[Dict[str, Any]] = []
    indexed_records = list(enumerate(payload))
    for normalized in parallel_map(
        indexed_records,
        lambda pair: normalize_grounding_record(
            record_index=pair[0],
            record=pair[1],
            drop_missing_images=drop_missing_images,
        ),
        workers=workers,
    ):
        if normalized is None:
            continue
        items.append(normalized)
    return items


def normalize_results_file(
    *,
    file_path: Path,
    results_dir: Path,
    drop_missing_images: bool,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    scene_id, viewpoint_id = infer_scene_and_viewpoint(file_path, results_dir)
    pano_path = build_realsee_pano_path(scene_id, viewpoint_id)
    mask_path = build_realsee_mask_path(scene_id, viewpoint_id)
    records = load_jsonl(file_path)
    for record_index, record in enumerate(records):
        normalized = normalize_canonical_sample_record(
            record=record,
            pano_path=pano_path,
            mask_path=mask_path,
            source_file=file_path,
            record_index=record_index,
            drop_missing_images=drop_missing_images,
        )
        if normalized is None:
            continue
        items.append(normalized)
    return items


def normalize_caption_record(
    *,
    record_index: int,
    record: Dict[str, Any],
    include_full_caption: bool,
    drop_missing_images: bool,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not isinstance(record, dict):
        return items
    remapped = remap_paths_in_payload(record)
    pano_path = normalize_optional_string(remapped.get("pano_path"))
    if not pano_path and drop_missing_images:
        return items
    description = normalize_optional_string(remapped.get("description"))
    scene_key = build_scene_key(remapped, record_index)

    if include_full_caption and description:
        items.append(
            {
                "id": f"caption_full:{scene_key}",
                "source": "caption_vqa",
                "task_family": "scene_caption",
                "subtask": "full_caption",
                "images": [pano_path] if pano_path else [],
                "messages": [
                    {"role": "user", "content": FULL_CAPTION_PROMPT},
                    {"role": "assistant", "content": description},
                ],
                "meta": {
                    "source_record_index": record_index,
                    "pano_path": pano_path,
                    "mask_path": normalize_optional_string(remapped.get("mask_path")),
                    "scene_key": scene_key,
                    "yaws_deg": remapped.get("yaws_deg"),
                },
            }
        )

    sections = parse_caption_sections(description)
    for section_name, prompt in CAPTION_SECTION_PROMPTS.items():
        answer = sections.get(section_name)
        if not answer:
            continue
        items.append(
            {
                "id": f"caption_{slugify(section_name)}:{scene_key}",
                "source": "caption_vqa",
                "task_family": "scene_caption",
                "subtask": slugify(section_name),
                "images": [pano_path] if pano_path else [],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {
                    "source_record_index": record_index,
                    "section": section_name,
                    "pano_path": pano_path,
                    "mask_path": normalize_optional_string(remapped.get("mask_path")),
                    "scene_key": scene_key,
                },
            }
        )
    return items


def normalize_grounding_record(
    *,
    record_index: int,
    record: Dict[str, Any],
    drop_missing_images: bool,
) -> Optional[Dict[str, Any]]:
    remapped = remap_paths_in_payload(record)
    normalized = normalize_generic_qa_record(
        record=remapped,
        source="replica_grounding",
        sample_id=f"grounding:{record_index:08d}",
        meta={"source_record_index": record_index},
        drop_missing_images=drop_missing_images,
    )
    if normalized is None:
        return None
    normalized["task_family"] = "grounding"
    normalized["subtask"] = "bfov_grounding"
    return normalized


def normalize_canonical_sample_record(
    *,
    record: Dict[str, Any],
    pano_path: str,
    mask_path: str,
    source_file: Path,
    record_index: int,
    drop_missing_images: bool = False,
) -> Optional[Dict[str, Any]]:
    messages = extract_messages(record)
    if not messages:
        return None
    images = [pano_path] if pano_path else []
    if drop_missing_images and not images:
        return None

    return {
        "id": str(record.get("sample_id", f"results:{record_index:08d}")),
        "source": "results_final_v2",
        "task_family": str(record.get("task_family", "generic_qa")),
        "subtask": str(record.get("generation_mode", record.get("task_family", "default"))),
        "images": images,
        "messages": messages,
        "meta": {
            "scene_id": record.get("scene_id"),
            "source_file": str(source_file),
            "source_record_index": record_index,
            "derived_pano_path": pano_path,
            "derived_mask_path": mask_path,
            "postprocess_disposition": record.get("postprocess_disposition"),
            "postprocess_job_id": record.get("postprocess_job_id"),
            "extra_fields": remap_paths_in_payload(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"messages", "sample_id", "scene_id", "task_family", "canonical_question", "canonical_answer"}
                }
            ),
        },
    }


def normalize_generic_qa_record(
    *,
    record: Dict[str, Any],
    source: str,
    sample_id: str,
    meta: Dict[str, Any],
    drop_missing_images: bool = False,
) -> Optional[Dict[str, Any]]:
    messages = extract_messages(record)
    images = extract_images(record)
    if not messages:
        return None
    if drop_missing_images and not images:
        return None

    item = {
        "id": sample_id,
        "source": source,
        "task_family": record.get("task_family", "generic_qa"),
        "subtask": record.get("subtask", "default"),
        "images": images,
        "messages": messages,
        "meta": build_meta(record, meta),
    }
    return item


def extract_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(record.get("messages"), list):
        return _normalize_message_list(record["messages"])

    if isinstance(record.get("conversations"), list):
        normalized: List[Dict[str, str]] = []
        for message in record["conversations"]:
            role = str(message.get("from", message.get("role", ""))).strip().lower()
            content = str(message.get("value", message.get("content", ""))).strip()
            mapped_role = _map_role(role)
            if mapped_role and content:
                normalized.append({"role": mapped_role, "content": content})
        return normalized

    question = first_nonempty(
        record,
        [
            "canonical_question",
            "question",
            "query",
            "prompt",
            "instruction",
            "input",
            "user",
            "caption_question",
            "template_question",
        ],
    )
    answer = first_nonempty(
        record,
        [
            "answer_text",
            "canonical_answer",
            "answer",
            "response",
            "output",
            "assistant",
            "target",
            "label",
            "caption_answer",
            "template_answer",
        ],
    )
    if question and answer:
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    return []


def extract_images(record: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    if isinstance(record.get("images"), list):
        candidates.extend([str(item) for item in record["images"] if item])

    for key in ["image", "image_path", "img", "img_path", "pano_path"]:
        value = record.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)

    if isinstance(record.get("image_paths"), list):
        candidates.extend([str(item) for item in record["image_paths"] if item])

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        candidate = remap_path(candidate)
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def build_meta(record: Dict[str, Any], seed_meta: Dict[str, Any]) -> Dict[str, Any]:
    ignored_keys = {
        "messages",
        "conversations",
        "images",
        "image",
        "image_path",
        "image_paths",
        "img",
        "img_path",
        "pano_path",
        "question",
        "query",
        "prompt",
        "instruction",
        "input",
        "user",
        "answer",
        "response",
        "output",
        "assistant",
        "target",
        "label",
    }
    extra = {key: remap_paths_in_payload(value) for key, value in record.items() if key not in ignored_keys}
    meta = dict(seed_meta)
    if extra:
        meta["extra_fields"] = extra
    return meta


def build_scene_key(record: Dict[str, Any], record_index: int) -> str:
    pano_path = str(record.get("pano_path", ""))
    if pano_path:
        path = Path(pano_path)
        parts = [part for part in path.parts if part.startswith("scene_")]
        if parts:
            scene_part = parts[-1]
            view_id = path.parent.name
            return f"{scene_part}_{view_id}"
        return path.stem
    return f"record_{record_index:06d}"


def infer_scene_and_viewpoint(file_path: Path, results_dir: Path) -> tuple[str, str]:
    relative = file_path.relative_to(results_dir)
    if len(relative.parts) < 3:
        raise ValueError(f"unexpected canonical_samples.jsonl layout: {file_path}")
    scene_id = relative.parts[0]
    viewpoint_id = relative.parts[1]
    return scene_id, viewpoint_id


def build_realsee_pano_path(scene_id: str, viewpoint_id: str) -> str:
    return f"{REALSEE_PUBLIC_ROOT}/{scene_id}/viewpoints/{viewpoint_id}/panoImage_1600.jpg"


def build_realsee_mask_path(scene_id: str, viewpoint_id: str) -> str:
    return f"{REALSEE_PUBLIC_ROOT}/{scene_id}/viewpoints/{viewpoint_id}/pano_mask.png"


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def first_nonempty(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        normalized = normalize_optional_string(value)
        if normalized:
            return normalized
    return ""


def normalize_optional_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)

    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        return [payload]
    return []


def _normalize_message_list(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = _map_role(str(message.get("role", message.get("from", ""))).strip().lower())
        content = str(message.get("content", message.get("value", ""))).strip()
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized


def _map_role(role: str) -> str:
    role_map = {
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "system": "system",
    }
    return role_map.get(role, "")


def parallel_map(items: List[Any], fn: Any, workers: int) -> List[Any]:
    if not items:
        return []
    worker_count = max(1, min(workers, len(items)))
    if worker_count == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fn, items))
