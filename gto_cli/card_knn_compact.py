from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .card_classifier import DEFAULT_MODEL_PATH, RANK_LABELS, SUIT_LABELS, glyph_feature, load_cv, prepare_glyph_image


def compact_card_classifier(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_model: Path,
    benchmark_rows_csvs: list[Path],
    top_per_sample: int = 3,
    min_per_label: int = 96,
    max_per_label: int = 256,
) -> dict[str, Any]:
    _cv2, np = load_cv()
    source = np.load(str(model_path), allow_pickle=False)
    metadata = json.loads(str(source["metadata"]))
    rank_features = source["rank_features"].astype(np.float32)
    rank_labels = source["rank_labels"]
    suit_features = source["suit_features"].astype(np.float32)
    suit_labels = source["suit_labels"]
    benchmark_rows = load_benchmark_rows(benchmark_rows_csvs)

    rank_keep, rank_stats = select_kind_indices(
        features=rank_features,
        labels=rank_labels,
        labels_order=RANK_LABELS,
        rows=benchmark_rows,
        kind="rank",
        top_per_sample=top_per_sample,
        min_per_label=min_per_label,
        max_per_label=max_per_label,
    )
    suit_keep, suit_stats = select_kind_indices(
        features=suit_features,
        labels=suit_labels,
        labels_order=SUIT_LABELS,
        rows=benchmark_rows,
        kind="suit",
        top_per_sample=top_per_sample,
        min_per_label=min_per_label,
        max_per_label=max_per_label,
    )

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    compact_metadata = {
        **metadata,
        "kind": "card_glyph_knn_compact",
        "source_model": str(model_path),
        "benchmark_rows_csvs": [str(path) for path in benchmark_rows_csvs],
        "top_per_sample": int(top_per_sample),
        "min_per_label": int(min_per_label),
        "max_per_label": int(max_per_label),
        "original_rank_feature_count": int(rank_features.shape[0]),
        "original_suit_feature_count": int(suit_features.shape[0]),
        "rank_feature_count": int(len(rank_keep)),
        "suit_feature_count": int(len(suit_keep)),
        "rank_compaction": rank_stats,
        "suit_compaction": suit_stats,
    }
    np.savez_compressed(
        str(output_model),
        rank_features=rank_features[rank_keep].astype(np.float32),
        rank_labels=rank_labels[rank_keep],
        suit_features=suit_features[suit_keep].astype(np.float32),
        suit_labels=suit_labels[suit_keep],
        metadata=json.dumps(compact_metadata, ensure_ascii=False),
    )
    return {
        "ok": True,
        "source_model": str(model_path),
        "model": str(output_model),
        "benchmark_rows": len(benchmark_rows),
        "metadata": compact_metadata,
    }


def load_benchmark_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    return rows


def select_kind_indices(
    *,
    features: Any,
    labels: Any,
    labels_order: tuple[str, ...],
    rows: list[dict[str, Any]],
    kind: str,
    top_per_sample: int,
    min_per_label: int,
    max_per_label: int,
) -> tuple[Any, dict[str, Any]]:
    cv2, np = load_cv()
    del cv2
    selected_by_label: dict[str, list[int]] = {label: [] for label in labels_order}
    selected_sets: dict[str, set[int]] = {label: set() for label in labels_order}
    sample_count_by_label: dict[str, int] = {label: 0 for label in labels_order}
    label_array = labels.astype(str)
    path_field = f"{kind}_path"
    expected_field = f"expected_{kind}"

    for row in rows:
        expected = str(row.get(expected_field) or "").strip()
        path_text = str(row.get(path_field) or "").strip()
        if expected not in selected_by_label or not path_text:
            continue
        image_path = Path(path_text)
        if not image_path.exists():
            continue
        image = cv2_read_gray(image_path)
        if image is None:
            continue
        feature = glyph_feature(prepare_glyph_image(image, kind), kind)
        label_indices = np.where(label_array == expected)[0]
        if label_indices.size == 0:
            continue
        sample_count_by_label[expected] += 1
        scores = features[label_indices] @ feature
        order = label_indices[np.argsort(-scores)[: max(1, int(top_per_sample))]]
        append_unique(selected_by_label[expected], selected_sets[expected], [int(idx) for idx in order])

    for label in labels_order:
        label_indices = np.where(label_array == label)[0]
        if label_indices.size == 0:
            continue
        existing = selected_by_label[label]
        existing_set = selected_sets[label]
        target = min(max(0, int(max_per_label)), max(int(min_per_label), len(existing)))
        target = min(target, int(label_indices.size))
        if len(existing) < target:
            add_diverse_indices(
                features=features,
                candidate_indices=[int(idx) for idx in label_indices],
                selected=existing,
                selected_set=existing_set,
                target=target,
            )
        if len(existing) > max_per_label:
            selected_by_label[label] = existing[: int(max_per_label)]
            selected_sets[label] = set(selected_by_label[label])

    keep = sorted({idx for indices in selected_by_label.values() for idx in indices})
    stats = {
        "original_feature_count": int(features.shape[0]),
        "feature_count": len(keep),
        "labels": {
            label: {
                "samples": sample_count_by_label[label],
                "kept": len(selected_by_label[label]),
                "available": int((label_array == label).sum()),
            }
            for label in labels_order
        },
    }
    return np.asarray(keep, dtype=np.int64), stats


def cv2_read_gray(path: Path) -> Any | None:
    cv2, _np = load_cv()
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def append_unique(target: list[int], seen: set[int], values: list[int]) -> None:
    for value in values:
        if value in seen:
            continue
        target.append(value)
        seen.add(value)


def add_diverse_indices(
    *,
    features: Any,
    candidate_indices: list[int],
    selected: list[int],
    selected_set: set[int],
    target: int,
) -> None:
    _cv2, np = load_cv()
    if not candidate_indices:
        return
    if not selected:
        selected.append(candidate_indices[0])
        selected_set.add(candidate_indices[0])
    while len(selected) < target:
        remaining = [idx for idx in candidate_indices if idx not in selected_set]
        if not remaining:
            break
        selected_features = features[selected]
        remaining_features = features[remaining]
        similarities = remaining_features @ selected_features.T
        diversity_scores = 1.0 - similarities.max(axis=1)
        best = remaining[int(np.argmax(diversity_scores))]
        selected.append(best)
        selected_set.add(best)


def format_compact_card_classifier_summary(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    return "\n".join(
        [
            f"Model: {payload.get('model')}",
            f"Source: {payload.get('source_model')}",
            f"Benchmark rows: {payload.get('benchmark_rows')}",
            (
                "Rank features: "
                f"{metadata.get('original_rank_feature_count')} -> {metadata.get('rank_feature_count')}"
            ),
            (
                "Suit features: "
                f"{metadata.get('original_suit_feature_count')} -> {metadata.get('suit_feature_count')}"
            ),
            f"top_per_sample={metadata.get('top_per_sample')} min_per_label={metadata.get('min_per_label')} max_per_label={metadata.get('max_per_label')}",
        ]
    )
