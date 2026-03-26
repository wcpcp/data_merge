from __future__ import annotations

import re
from typing import Dict


HEADER_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_caption_sections(description: str) -> Dict[str, str]:
    if not description:
        return {}

    matches = list(HEADER_PATTERN.finditer(description))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for index, match in enumerate(matches):
        raw_title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(description)
        body = _clean_section_body(description[start:end])
        title = canonicalize_heading(raw_title)
        if title and body:
            sections[title] = body
    return sections


def canonicalize_heading(raw_title: str) -> str:
    upper = raw_title.upper()
    candidates = [
        "GLOBAL LAYOUT",
        "TOPOLOGICAL RELATIONS",
        "NAVIGABLE FRONTIERS",
        "PHYSICAL CONSTRAINTS",
        "SPATIAL REASONING SUMMARY",
        "FINAL RECONSTRUCTION",
    ]
    for candidate in candidates:
        if candidate in upper:
            return candidate
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 /_-]", " ", upper)).strip()


def _clean_section_body(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(---\s*)+", "", cleaned)
    cleaned = re.sub(r"(\s*---)+$", "", cleaned)
    return cleaned.strip()
