from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .card_classifier import RANK_LABELS, SUIT_LABELS, parse_card_label


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_CARD_REPO = "F1NN21/playing-cards"
DEFAULT_HF_CARD_DIR = PROJECT_ROOT / "pict" / "card_datasets" / "hf_f1nn21_playing_cards"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_CARD_LABELS = tuple(f"{rank}{suit}" for rank in RANK_LABELS for suit in SUIT_LABELS)


def download_card_dataset(
    *,
    repo_id: str = DEFAULT_HF_CARD_REPO,
    output_dir: Path = DEFAULT_HF_CARD_DIR,
    repo_type: str = "dataset",
    allow_patterns: list[str] | None = None,
    refresh: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and not refresh:
        summary = dataset_summary(
            output_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            downloaded=False,
            reason="existing_output_dir",
        )
        (output_dir / "dataset_download_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required: pip install huggingface_hub") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns or None,
        force_download=bool(refresh),
        local_files_only=bool(local_files_only),
    )
    summary = dataset_summary(
        Path(path),
        repo_id=repo_id,
        repo_type=repo_type,
        downloaded=True,
        reason="snapshot_download",
    )
    summary["requested_output_dir"] = str(output_dir)
    summary["allow_patterns"] = allow_patterns or []
    summary["local_files_only"] = bool(local_files_only)
    (output_dir / "dataset_download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def dataset_summary(
    root: Path,
    *,
    repo_id: str,
    repo_type: str,
    downloaded: bool,
    reason: str,
) -> dict[str, Any]:
    root = Path(root)
    image_paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS) if root.exists() else []
    label_dirs = []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()) if root.exists() else []:
        image_count = sum(1 for child in directory.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS)
        if image_count:
            label_dirs.append({"dir": str(directory), "images": image_count})
    likely_roots = sorted(
        {
            str(Path(item["dir"]).parent)
            for item in label_dirs
            if Path(item["dir"]).parent != root.parent
        }
    )
    card_coverage = summarize_card_coverage(image_paths)
    return {
        "ok": True,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "output_dir": str(root),
        "downloaded": bool(downloaded),
        "reason": reason,
        "image_count": len(image_paths),
        "label_dir_count": len(label_dirs),
        "likely_dataset_roots": likely_roots[:20],
        "label_dir_examples": label_dirs[:20],
        "card_coverage": card_coverage,
    }


def summarize_card_coverage(image_paths: list[Path]) -> dict[str, Any]:
    counts = {card: 0 for card in EXPECTED_CARD_LABELS}
    unparsed: list[str] = []
    for path in image_paths:
        parsed = parse_card_label(path)
        if parsed is None:
            unparsed.append(str(path))
            continue
        rank, suit = parsed
        card = f"{rank}{suit}"
        if card in counts:
            counts[card] += 1
        else:
            unparsed.append(str(path))
    missing = [card for card in EXPECTED_CARD_LABELS if counts.get(card, 0) <= 0]
    duplicates = {card: count for card, count in counts.items() if count > 1}
    rank_counts = {rank: sum(counts[f"{rank}{suit}"] for suit in SUIT_LABELS) for rank in RANK_LABELS}
    suit_counts = {suit: sum(counts[f"{rank}{suit}"] for rank in RANK_LABELS) for suit in SUIT_LABELS}
    return {
        "parsed_card_count": sum(counts.values()),
        "expected_card_count": len(EXPECTED_CARD_LABELS),
        "complete_deck": not missing and not duplicates,
        "missing_cards": missing,
        "duplicate_cards": duplicates,
        "unparsed_count": len(unparsed),
        "unparsed_examples": unparsed[:20],
        "rank_counts": rank_counts,
        "suit_counts": suit_counts,
    }


def format_dataset_download_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"download-card-dataset failed: {payload.get('error')}"
    roots = payload.get("likely_dataset_roots") or []
    lines = [
        f"Repo: {payload.get('repo_id')}",
        f"Output: {payload.get('output_dir')}",
        f"Downloaded: {payload.get('downloaded')} ({payload.get('reason')})",
        f"Images: {payload.get('image_count', 0)}",
        f"Label dirs: {payload.get('label_dir_count', 0)}",
    ]
    coverage = payload.get("card_coverage") or {}
    if coverage:
        lines.extend(
            [
                f"Parsed cards: {coverage.get('parsed_card_count', 0)}/{coverage.get('expected_card_count', 52)}",
                f"Complete deck: {coverage.get('complete_deck')}",
                f"Missing cards: {coverage.get('missing_cards') or []}",
                f"Duplicate cards: {coverage.get('duplicate_cards') or {}}",
                f"Unparsed images: {coverage.get('unparsed_count', 0)}",
            ]
        )
    if roots:
        lines.append(f"Likely dataset root: {roots[0]}")
    return "\n".join(lines)
