from __future__ import annotations

import csv
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
        "source_name": "hos_sft",
        "display_name": "Thinking in 360 / HOS-SFT",
        "jsonl_url": "https://huggingface.co/datasets/humanoid-vstar/hos-sft/resolve/main/hos_sft_sharegpt.jsonl?download=true",
        "zip_url": "https://huggingface.co/datasets/humanoid-vstar/hos-sft/resolve/main/hos_sft_sharegpt.zip?download=true",
        "jsonl_filename": "hos_sft_sharegpt.jsonl",
        "zip_filename": "hos_sft_sharegpt.zip",
        "relative_root": "hos_sft_sharegpt",
    },
    {
        "source_name": "hps_sft",
        "display_name": "Thinking in 360 / HPS-SFT",
        "jsonl_url": "https://huggingface.co/datasets/humanoid-vstar/hps-sft/resolve/main/hps_sft_sharegpt.jsonl?download=true",
        "zip_url": "https://huggingface.co/datasets/humanoid-vstar/hps-sft/resolve/main/hps_sft_sharegpt.zip?download=true",
        "jsonl_filename": "hps_sft_sharegpt.jsonl",
        "zip_filename": "hps_sft_sharegpt.zip",
        "relative_root": "hps_sft_sharegpt",
    },
)

PANOENV_DATASET_CANDIDATES = ("7zkk/PanoEnv", "guangmulizi/PanoEnv")

OSR_BENCH_NOTES = {
    "trainable": True,
    "benchmark_safe_for_same_benchmark": False,
    "summary": (
        "OSR-Bench ships official QA pairs and images, so it can be converted into multimodal SFT data. "
        "However, the public release does not expose a clean training split, so training on all samples will "
        "contaminate OSR-Bench evaluation unless you create your own holdout."
    ),
    "official_sources": [
        "https://huggingface.co/datasets/UUUserna/OSR-Bench",
        OSR_BENCH_QA_URL,
    ],
}

THINKING_IN_360_NOTES = {
    "trainable": True,
    "benchmark_safe_for_same_benchmark": True,
    "summary": (
        "Use the official HOS-SFT and HPS-SFT releases for training. They are already SFT corpora. "
        "Do not train on HSTAR-Bench if you want a clean benchmark result."
    ),
    "official_sources": [
        "https://humanoid-vstar.github.io/",
        "https://huggingface.co/datasets/humanoid-vstar/hos-sft",
        "https://huggingface.co/datasets/humanoid-vstar/hps-sft",
    ],
}

PANOENV_NOTES = {
    "trainable": True,
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
    include_osr_bench: bool = True
    include_thinking_in_360: bool = True
    include_panoenv: bool = True


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

    if config.include_osr_bench:
        osr_items = build_osr_bench_training_items(config.cache_dir, config.max_osr_records)
        osr_path = config.output_dir / "osr_bench_training_multimodal_blocks.json"
        _write_json(osr_path, osr_items)
        stats["files"]["osr_bench"] = str(osr_path)
        stats["counts"]["osr_bench"] = len(osr_items)

    if config.include_thinking_in_360:
        thinking_items = build_thinking_in_360_training_items(config.cache_dir, config.max_thinking_records)
        thinking_path = config.output_dir / "thinking_in_360_training_multimodal_blocks.json"
        _write_json(thinking_path, thinking_items)
        stats["files"]["thinking_in_360"] = str(thinking_path)
        stats["counts"]["thinking_in_360"] = len(thinking_items)

    if config.include_panoenv:
        panoenv_items, dataset_name = build_panoenv_training_items(config.cache_dir, config.max_panoenv_records)
        panoenv_path = config.output_dir / "panoenv_training_multimodal_blocks.json"
        _write_json(panoenv_path, panoenv_items)
        stats["files"]["panoenv"] = str(panoenv_path)
        stats["counts"]["panoenv"] = len(panoenv_items)
        stats["notes"]["panoenv"]["resolved_dataset"] = dataset_name

    stats_path = config.output_dir / "external_benchmark_stats.json"
    stats["files"]["stats"] = str(stats_path)
    _write_json(stats_path, stats)
    return stats


def build_osr_bench_training_items(cache_dir: Path, max_records: Optional[int]) -> List[Dict[str, Any]]:
    dataset_cache = cache_dir / "osr_bench"
    dataset_cache.mkdir(parents=True, exist_ok=True)
    qa_csv_path = dataset_cache / "qa.csv"
    _download_to_path(OSR_BENCH_QA_URL, qa_csv_path)

    items: List[Dict[str, Any]] = []
    with qa_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if max_records is not None and len(items) >= max_records:
                break
            image_id = str(row.get("image_id", "")).strip()
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not image_id or not question or not answer:
                continue
            image_path = dataset_cache / image_id
            _download_to_path(f"{OSR_BENCH_IMAGE_BASE_URL}/{image_id}?download=true", image_path)
            items.append(
                build_single_turn_training_item(
                    record_id=f"osr_bench:{safe_id(Path(image_id).with_suffix('').as_posix())}:{row.get('turn_id', row_index)}",
                    question=question,
                    answer=answer,
                    image_paths=[image_path],
                    system_prompt=ERP_MULTIMODAL_SYSTEM_PROMPT,
                )
            )
    return items


def build_thinking_in_360_training_items(cache_dir: Path, max_records: Optional[int]) -> List[Dict[str, Any]]:
    dataset_cache = cache_dir / "thinking_in_360"
    dataset_cache.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for source in THINKING_IN_360_SOURCES:
        if max_records is not None and len(items) >= max_records:
            break

        jsonl_path = dataset_cache / source["jsonl_filename"]
        zip_path = dataset_cache / source["zip_filename"]
        extract_dir = dataset_cache / source["source_name"]

        _download_to_path(str(source["jsonl_url"]), jsonl_path)
        _download_to_path(str(source["zip_url"]), zip_path)
        _extract_zip_once(zip_path, extract_dir)

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if max_records is not None and len(items) >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                item = build_thinking_record(
                    row=row,
                    record_index=row_index,
                    source_name=str(source["source_name"]),
                    extract_dir=extract_dir,
                )
                if item is not None:
                    items.append(item)
    return items


def build_panoenv_training_items(cache_dir: Path, max_records: Optional[int]) -> Tuple[List[Dict[str, Any]], str]:
    dataset_name, train_split = load_panoenv_train_split()
    dataset_cache = cache_dir / "panoenv"
    image_cache = dataset_cache / "images"
    image_cache.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for row in train_split:
        if max_records is not None and len(items) >= max_records:
            break
        image_path = save_panoenv_row_image(row, image_cache)
        row_items = build_panoenv_training_items_from_row(row, image_path)
        if max_records is not None:
            remaining = max_records - len(items)
            row_items = row_items[:remaining]
        items.extend(row_items)

    return items, dataset_name


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


def build_thinking_record(
    row: Dict[str, Any],
    record_index: int,
    source_name: str,
    extract_dir: Path,
) -> Optional[Dict[str, Any]]:
    conversations = row.get("conversations", [])
    if not isinstance(conversations, list) or not conversations:
        return None

    raw_images = row.get("images", [])
    if not isinstance(raw_images, list) or not raw_images:
        return None

    resolved_images: List[Path] = []
    for raw_image in raw_images:
        resolved = resolve_extracted_image_path(extract_dir, str(raw_image))
        if resolved is None:
            raise FileNotFoundError(f"missing extracted image for Thinking in 360 sample: {raw_image}")
        resolved_images.append(resolved)

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


def build_panoenv_training_items_from_row(row: Dict[str, Any], image_path: Path) -> List[Dict[str, Any]]:
    env_name = str(row.get("env", "")).strip()
    image_id = str(row.get("image_id", "")).strip()
    questions = row.get("questions", [])
    if not env_name or not image_id or not isinstance(questions, list):
        return []

    items: List[Dict[str, Any]] = []
    for question_entry in questions:
        if not isinstance(question_entry, dict):
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


def resolve_extracted_image_path(extract_dir: Path, relative_path: str) -> Optional[Path]:
    relative = Path(relative_path)
    direct = extract_dir / relative
    if direct.exists():
        return direct
    nested = extract_dir / relative.name
    if nested.exists():
        return nested
    return None


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
