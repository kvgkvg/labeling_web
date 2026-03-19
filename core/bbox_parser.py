"""
Parse messy extracted_bbox strings into list of {x1, y1, x2, y2, label} dicts.

Handled formats:
  "[]"                                       → []
  '["[10, 10, 498, 200]"]'                   → no label
  '["[300,700,430,735] {LEV MZ 318}"]'       → with label
  '[[300, 150, 600, 230]]'                   → nested list format
"""

import ast
import json
import re
from typing import Any


def _parse_coords(raw: str) -> tuple[list[float], str]:
    """Extract [x1,y1,x2,y2] and optional {label} from a bbox entry string."""
    raw = raw.strip()
    label = ""
    label_match = re.search(r"\{([^}]*)\}", raw)
    if label_match:
        label = label_match.group(1).strip()
        raw = raw[: label_match.start()].strip()

    # Extract numbers
    nums = re.findall(r"[-+]?\d*\.?\d+", raw)
    if len(nums) < 4:
        return [], label
    coords = [float(n) for n in nums[:4]]
    return coords, label


def _make_bbox(coords: list[float], label: str = "") -> dict:
    x1, y1, x2, y2 = coords
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label}


def parse_bbox(raw_value: Any) -> list[dict]:
    """
    Parse any bbox representation into a list of bbox dicts.
    Returns [] on failure or empty input.
    """
    if raw_value is None:
        return []

    # Already a list (rare — if caller passes pre-parsed data)
    if isinstance(raw_value, list):
        if not raw_value:
            return []
        result = []
        for item in raw_value:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                result.append(_make_bbox([float(v) for v in item[:4]]))
            elif isinstance(item, dict):
                result.append({
                    "x1": float(item.get("x1", 0)),
                    "y1": float(item.get("y1", 0)),
                    "x2": float(item.get("x2", 0)),
                    "y2": float(item.get("y2", 0)),
                    "label": item.get("label", ""),
                })
            elif isinstance(item, str):
                coords, label = _parse_coords(item)
                if len(coords) == 4:
                    result.append(_make_bbox(coords, label))
        return result

    raw_str = str(raw_value).strip()

    # Empty
    if raw_str in ("[]", "", "None", "nan"):
        return []

    # Try JSON parse first
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, list):
            if not parsed:
                return []
            result = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 4:
                    # [[300, 150, 600, 230]] format
                    result.append(_make_bbox([float(v) for v in item[:4]]))
                elif isinstance(item, str):
                    # '["[10, 10, 498, 200]"]' or '["[300,700,430,735] {label}"]'
                    coords, label = _parse_coords(item)
                    if len(coords) == 4:
                        result.append(_make_bbox(coords, label))
                elif isinstance(item, dict):
                    result.append({
                        "x1": float(item.get("x1", 0)),
                        "y1": float(item.get("y1", 0)),
                        "x2": float(item.get("x2", 0)),
                        "y2": float(item.get("y2", 0)),
                        "label": item.get("label", ""),
                    })
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try ast.literal_eval as fallback (handles Python-style lists)
    try:
        parsed = ast.literal_eval(raw_str)
        return parse_bbox(parsed)  # recurse with proper list
    except (ValueError, SyntaxError):
        pass

    # Last resort: extract all [x,y,x,y] patterns with optional {label}
    result = []
    pattern = re.compile(r"\[([^\[\]]+)\](?:\s*\{([^}]*)\})?")
    for m in pattern.finditer(raw_str):
        nums = re.findall(r"[-+]?\d*\.?\d+", m.group(1))
        if len(nums) >= 4:
            coords = [float(n) for n in nums[:4]]
            label = (m.group(2) or "").strip()
            result.append(_make_bbox(coords, label))
    return result
