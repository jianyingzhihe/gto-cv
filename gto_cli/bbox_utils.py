from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_BBOX_FILE = Path("video_frames") / "screen_calibrate" / "bbox.json"
PLACEHOLDER_BBOX = "x,y,w,h"


def parse_bbox_values(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [part.strip() for part in str(text).replace(",", " ").split() if part.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be four numbers: x,y,width,height")
    try:
        left, top, width, height = (int(float(part)) for part in parts)
    except ValueError as error:
        raise ValueError(
            "--bbox must be numeric x,y,width,height. Use screen-cv --pick-bbox first, "
            "or pass --bbox-file video_frames\\screen_calibrate\\bbox.json."
        ) from error
    if width <= 0 or height <= 0:
        raise ValueError("--bbox width and height must be positive")
    return left, top, width, height


def bbox_values_to_text(values: tuple[int, int, int, int] | list[int | float]) -> str:
    parsed = parse_bbox_values(",".join(str(value) for value in values))
    if parsed is None:
        raise ValueError("bbox values are empty")
    return ",".join(str(value) for value in parsed)


def load_bbox_text(path: Path) -> str:
    payload = load_bbox_payload(path)
    return bbox_payload_to_text(payload, source=path)


def load_bbox_payload(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"bbox file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def load_outer_bbox_text(path: Path) -> str | None:
    """Return the canonical manually selected full poker-client region.

    A reviewed ``analysis_bbox.json`` holds only the inner table coordinates.
    Its sibling ``bbox.json`` is the single source of truth for the full
    poker-client capture used by action-control recognition.  The embedded
    ``outer_region`` fallback exists only for legacy reviewed files whose
    original manual bbox no longer exists.
    """

    path = Path(path)
    manual_bbox = path.with_name("bbox.json")
    if path.name.lower() == "analysis_bbox.json" and manual_bbox.is_file():
        return load_bbox_text(manual_bbox)

    payload = load_bbox_payload(path)
    if not isinstance(payload, dict):
        return None
    outer_region = payload.get("outer_region")
    if not isinstance(outer_region, dict):
        return None
    return bbox_payload_to_text(outer_region, source=Path(path))


def load_rebased_analysis_bbox_text(manual_bbox_path: Path, *, max_aspect_ratio_change: float = 0.08) -> str | None:
    """Project a reviewed inner table box into a newly selected full client box."""

    manual_bbox_path = Path(manual_bbox_path)
    if manual_bbox_path.name.lower() != "bbox.json" or not manual_bbox_path.is_file():
        return None
    reviewed_path = manual_bbox_path.with_name("analysis_bbox.json")
    if not reviewed_path.is_file():
        return None

    current_outer = parse_bbox_values(load_bbox_text(manual_bbox_path))
    reviewed_payload = load_bbox_payload(reviewed_path)
    if current_outer is None or not isinstance(reviewed_payload, dict):
        return None
    reference_outer = reviewed_payload.get("outer_reference") or reviewed_payload.get("outer_region")
    if not isinstance(reference_outer, dict):
        return None
    reference_text = bbox_payload_to_text(reference_outer, source=reviewed_path)
    reference = parse_bbox_values(reference_text)
    if reference is None or reference[2] <= 0 or reference[3] <= 0:
        return None

    current_aspect = current_outer[2] / current_outer[3]
    reference_aspect = reference[2] / reference[3]
    if abs(current_aspect / reference_aspect - 1.0) > max(0.0, float(max_aspect_ratio_change)):
        return None

    relative = reviewed_payload.get("relative_to_outer")
    if not isinstance(relative, dict):
        inner = parse_bbox_values(bbox_payload_to_text(reviewed_payload, source=reviewed_path))
        if inner is None:
            return None
        relative = {
            "x": (inner[0] - reference[0]) / reference[2],
            "y": (inner[1] - reference[1]) / reference[3],
            "width": inner[2] / reference[2],
            "height": inner[3] / reference[3],
        }

    try:
        x = float(relative["x"])
        y = float(relative["y"])
        width = float(relative["width"])
        height = float(relative["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.0 or y + height > 1.0:
        return None

    left = current_outer[0] + round(x * current_outer[2])
    top = current_outer[1] + round(y * current_outer[3])
    projected_width = max(1, round(width * current_outer[2]))
    projected_height = max(1, round(height * current_outer[3]))
    if left + projected_width > current_outer[0] + current_outer[2]:
        return None
    if top + projected_height > current_outer[1] + current_outer[3]:
        return None
    return bbox_values_to_text((left, top, projected_width, projected_height))


def reviewed_bbox_requires_refresh(path: Path) -> bool:
    """Return whether a reviewed inner table crop predates its manual outer crop.

    ``bbox.json`` is written each time the user redraws the complete poker
    client.  ``analysis_bbox.json`` is the second, reviewed inner-table crop.
    Running a stale reviewed crop mixes a new full window with old inner
    coordinates, so callers must ask for another review instead of guessing.
    """

    path = Path(path)
    if path.name.lower() != "analysis_bbox.json" or not path.is_file():
        return False
    manual_bbox = path.with_name("bbox.json")
    if not manual_bbox.is_file():
        return False
    return manual_bbox.stat().st_mtime_ns > path.stat().st_mtime_ns


def bbox_payload_to_text(payload: Any, *, source: Path | None = None) -> str:
    if isinstance(payload, str):
        return bbox_values_to_text([part for part in payload.replace(",", " ").split() if part])
    if isinstance(payload, (list, tuple)):
        if len(payload) != 4:
            raise ValueError(f"bbox file must contain four values: {source or ''}".strip())
        return bbox_values_to_text(list(payload))
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported bbox file format: {source or ''}".strip())

    for key in ("text", "bbox_text"):
        value = payload.get(key)
        if value:
            return bbox_values_to_text([part for part in str(value).replace(",", " ").split() if part])

    for key in ("bbox", "values"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return bbox_values_to_text(list(value))
        if isinstance(value, str) and value.strip():
            return bbox_values_to_text([part for part in value.replace(",", " ").split() if part])

    key_sets = (
        ("left", "top", "width", "height"),
        ("x", "y", "w", "h"),
        ("x", "y", "width", "height"),
    )
    for keys in key_sets:
        if all(key in payload for key in keys):
            return bbox_values_to_text([payload[key] for key in keys])

    region = payload.get("region")
    if isinstance(region, dict):
        return bbox_payload_to_text(region, source=source)

    raise ValueError(f"bbox file does not contain text/left/top/width/height: {source or ''}".strip())


def find_latest_bbox_file(root: Path = Path("video_frames")) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    candidates = sorted(root.rglob("bbox.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_bbox_text(
    bbox: str | None,
    *,
    bbox_file: Path | None = None,
    latest_bbox: bool = False,
    latest_root: Path = Path("video_frames"),
) -> str:
    if bbox_file is not None:
        return load_bbox_text(Path(bbox_file))
    if latest_bbox:
        latest = find_latest_bbox_file(latest_root)
        if latest is None:
            raise ValueError(f"no bbox.json found under {latest_root}")
        return load_bbox_text(latest)
    if bbox:
        return str(bbox)
    return ""
