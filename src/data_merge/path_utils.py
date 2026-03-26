from __future__ import annotations

from typing import Any

from .config import PATH_REPLACEMENTS


def remap_path(value: str) -> str:
    for old_prefix, new_prefix in PATH_REPLACEMENTS:
        if value.startswith(old_prefix):
            return new_prefix + value[len(old_prefix) :]
    return value


def remap_paths_in_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return remap_path(payload)
    if isinstance(payload, list):
        return [remap_paths_in_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: remap_paths_in_payload(value) for key, value in payload.items()}
    return payload
