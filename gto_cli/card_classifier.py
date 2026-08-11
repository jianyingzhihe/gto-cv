from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "pict" / "card_templates"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pict" / "card_models" / "card_glyph_knn.npz"

RANK_LABELS = tuple("AKQJT98765432")
SUIT_LABELS = tuple("shdc")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

_MODEL_CACHE: dict[Path, dict[str, Any] | None] = {}
_GLYPH_CLASSIFY_CACHE: dict[tuple[Any, ...], dict[str, Any] | None] = {}
_GLYPH_CLASSIFY_CACHE_LIMIT = 8192


def train_card_classifier(
    *,
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    dataset_dirs: list[Path] | None = None,
    glyph_dirs: list[Path] | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    seed_model_path: Path | None = None,
    seed_conflict_policy: str = "manual_override",
    seed_guard: bool = False,
    seed_guard_rank_score: float = 0.55,
    seed_guard_rank_margin: float = 0.10,
    seed_guard_suit_score: float = 0.70,
    seed_guard_suit_margin: float = 0.04,
    include_templates: bool = True,
    augment: int = 8,
    external_augment: int | None = None,
    glyph_augment: int | None = None,
    max_external: int | None = None,
) -> dict[str, Any]:
    if seed_conflict_policy not in {"manual_override", "keep_seed"}:
        raise ValueError("seed_conflict_policy must be manual_override or keep_seed")
    cv2, np = load_cv()
    rank_samples: list[tuple[str, Any, str]] = []
    suit_samples: list[tuple[str, Any, str]] = []
    external_rank_samples: list[tuple[str, Any, str]] = []
    external_suit_samples: list[tuple[str, Any, str]] = []
    glyph_rank_samples: list[tuple[str, Any, str]] = []
    glyph_suit_samples: list[tuple[str, Any, str]] = []
    seed_rank_features = np.zeros((0, feature_length("rank")), np.float32)
    seed_rank_labels = np.array([], dtype="<U2")
    seed_suit_features = np.zeros((0, feature_length("suit")), np.float32)
    seed_suit_labels = np.array([], dtype="<U2")
    skipped: list[dict[str, str]] = []

    if seed_model_path is not None:
        seed_model_path = Path(seed_model_path)
        seed_model = load_card_classifier(seed_model_path) if seed_model_path.exists() else None
        if seed_model is None:
            skipped.append({"path": str(seed_model_path), "reason": "seed_model_missing_or_unreadable"})
        else:
            seed_rank_features = seed_model["rank_features"].astype(np.float32)
            seed_rank_labels = seed_model["rank_labels"]
            seed_suit_features = seed_model["suit_features"].astype(np.float32)
            seed_suit_labels = seed_model["suit_labels"]

    if include_templates:
        template_rank_samples, template_suit_samples = load_template_glyph_samples(template_dir)
        rank_samples.extend(template_rank_samples)
        suit_samples.extend(template_suit_samples)

    external_seen = 0
    for dataset_dir in dataset_dirs or []:
        for path in iter_image_files(dataset_dir):
            if max_external is not None and external_seen >= max_external:
                break
            label = parse_card_label(path)
            if label is None:
                skipped.append({"path": str(path), "reason": "label_not_found"})
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                skipped.append({"path": str(path), "reason": "image_read_failed"})
                continue
            rank, suit = label
            extracted = extract_corner_glyphs(image)
            if extracted is None:
                skipped.append({"path": str(path), "reason": "corner_glyph_extract_failed"})
                continue
            rank_image, suit_image = extracted
            external_rank_samples.append((rank, rank_image, str(path)))
            external_suit_samples.append((suit, suit_image, str(path)))
            external_seen += 1

    for glyph_dir in glyph_dirs or []:
        rank_glyphs, suit_glyphs, glyph_skipped = load_labeled_glyph_samples(Path(glyph_dir))
        glyph_rank_samples.extend(rank_glyphs)
        glyph_suit_samples.extend(suit_glyphs)
        skipped.extend(glyph_skipped)

    external_augment = augment if external_augment is None else int(external_augment)
    glyph_augment = augment if glyph_augment is None else int(glyph_augment)
    rank_features, rank_labels = merge_feature_tables(
        [
            (seed_rank_features, seed_rank_labels),
            make_feature_table(rank_samples, "rank", augment=augment),
            make_feature_table(external_rank_samples, "rank", augment=max(0, external_augment)),
            make_feature_table(glyph_rank_samples, "rank", augment=max(0, glyph_augment)),
        ],
        "rank",
        seed_conflict_policy=seed_conflict_policy,
    )
    suit_features, suit_labels = merge_feature_tables(
        [
            (seed_suit_features, seed_suit_labels),
            make_feature_table(suit_samples, "suit", augment=augment),
            make_feature_table(external_suit_samples, "suit", augment=max(0, external_augment)),
            make_feature_table(glyph_suit_samples, "suit", augment=max(0, glyph_augment)),
        ],
        "suit",
        seed_conflict_policy=seed_conflict_policy,
    )
    if rank_features.shape[0] == 0:
        raise ValueError("no rank training samples found")
    if suit_features.shape[0] == 0:
        raise ValueError("no suit training samples found")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "card_glyph_knn",
        "rank_source_count": len(rank_samples) + len(external_rank_samples) + len(glyph_rank_samples),
        "suit_source_count": len(suit_samples) + len(external_suit_samples) + len(glyph_suit_samples),
        "seed_model_path": str(seed_model_path) if seed_model_path is not None else "",
        "seed_conflict_policy": seed_conflict_policy,
        "seed_guard": bool(seed_guard),
        "seed_guard_rank_score": float(seed_guard_rank_score),
        "seed_guard_rank_margin": float(seed_guard_rank_margin),
        "seed_guard_suit_score": float(seed_guard_suit_score),
        "seed_guard_suit_margin": float(seed_guard_suit_margin),
        "seed_rank_feature_count": int(seed_rank_features.shape[0]),
        "seed_suit_feature_count": int(seed_suit_features.shape[0]),
        "local_rank_source_count": len(rank_samples),
        "local_suit_source_count": len(suit_samples),
        "external_rank_source_count": len(external_rank_samples),
        "external_suit_source_count": len(external_suit_samples),
        "glyph_rank_source_count": len(glyph_rank_samples),
        "glyph_suit_source_count": len(glyph_suit_samples),
        "rank_feature_count": int(rank_features.shape[0]),
        "suit_feature_count": int(suit_features.shape[0]),
        "rank_labels": sorted(set(rank_labels.tolist()), key=rank_sort_key),
        "suit_labels": sorted(set(suit_labels.tolist())),
        "template_dir": str(template_dir),
        "dataset_dirs": [str(path) for path in dataset_dirs or []],
        "glyph_dirs": [str(path) for path in glyph_dirs or []],
        "augment": int(augment),
        "external_augment": int(external_augment),
        "glyph_augment": int(glyph_augment),
        "skipped_count": len(skipped),
    }
    np.savez_compressed(
        str(model_path),
        rank_features=rank_features.astype(np.float32),
        rank_labels=rank_labels,
        suit_features=suit_features.astype(np.float32),
        suit_labels=suit_labels,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    _MODEL_CACHE.pop(model_path.resolve(), None)
    clear_glyph_classify_cache()
    return {
        "ok": True,
        "model": str(model_path),
        "metadata": metadata,
        "skipped_examples": skipped[:20],
    }


def load_template_glyph_samples(template_dir: Path) -> tuple[list[tuple[str, Any, str]], list[tuple[str, Any, str]]]:
    cv2, _np = load_cv()
    ranks: list[tuple[str, Any, str]] = []
    suits: list[tuple[str, Any, str]] = []
    if not template_dir.exists():
        return ranks, suits
    for path in sorted(template_dir.glob("rank_*.png")):
        label = path.stem.removeprefix("rank_").split("_", 1)[0]
        if label not in RANK_LABELS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            ranks.append((label, prepare_glyph_image(image, "rank"), str(path)))
    for path in sorted(template_dir.glob("suit_*.png")):
        label = path.stem.removeprefix("suit_").split("_", 1)[0]
        if label not in SUIT_LABELS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            suits.append((label, prepare_glyph_image(image, "suit"), str(path)))
    return ranks, suits


def load_labeled_glyph_samples(glyph_dir: Path) -> tuple[list[tuple[str, Any, str]], list[tuple[str, Any, str]], list[dict[str, str]]]:
    cv2, _np = load_cv()
    ranks: list[tuple[str, Any, str]] = []
    suits: list[tuple[str, Any, str]] = []
    skipped: list[dict[str, str]] = []
    glyph_dir = Path(glyph_dir)
    for kind, labels, output in (("rank", RANK_LABELS, ranks), ("suit", SUIT_LABELS, suits)):
        root = glyph_dir / kind
        if not root.exists():
            continue
        for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            label = label_dir.name
            if label not in labels:
                skipped.append({"path": str(label_dir), "reason": f"{kind}_label_not_allowed"})
                continue
            for path in sorted(item for item in label_dir.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    skipped.append({"path": str(path), "reason": f"{kind}_glyph_read_failed"})
                    continue
                output.append((label, prepare_glyph_image(image, kind), str(path)))
    return ranks, suits, skipped


def iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_card_label(path: Path) -> tuple[str, str] | None:
    rank_aliases = {
        "a": "A",
        "ace": "A",
        "k": "K",
        "king": "K",
        "q": "Q",
        "queen": "Q",
        "j": "J",
        "jack": "J",
        "t": "T",
        "0": "T",
        "10": "T",
        "ten": "T",
        "9": "9",
        "nine": "9",
        "8": "8",
        "eight": "8",
        "7": "7",
        "seven": "7",
        "6": "6",
        "six": "6",
        "5": "5",
        "five": "5",
        "4": "4",
        "four": "4",
        "3": "3",
        "three": "3",
        "2": "2",
        "two": "2",
    }
    suit_aliases = {
        "s": "s",
        "spade": "s",
        "spades": "s",
        "h": "h",
        "heart": "h",
        "hearts": "h",
        "d": "d",
        "diamond": "d",
        "diamonds": "d",
        "c": "c",
        "club": "c",
        "clubs": "c",
    }

    # A manually reviewed dataset often stores the corrected card as the parent
    # folder while preserving the old, wrong filename. Trust exact parent labels
    # before scanning noisy stems such as validation timestamps.
    for parent in path.parents[:4]:
        parsed_parent = parse_compact_card_token(parent.name, rank_aliases, suit_aliases)
        if parsed_parent is not None:
            return parsed_parent

    for bracket_token in re.findall(r"\[([0-9AQJKTCDHS]{2,3})\]", path.stem.upper()):
        parsed_bracket = parse_compact_card_token(bracket_token, rank_aliases, suit_aliases)
        if parsed_bracket is not None:
            return parsed_bracket

    pieces = [path.stem, *[parent.name for parent in path.parents[:3]]]
    text = " ".join(pieces).lower()
    text = (
        text.replace("♠", " spades ")
        .replace("♥", " hearts ")
        .replace("♦", " diamonds ")
        .replace("♣", " clubs ")
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = text.split()

    for word in words:
        parsed_word = parse_compact_card_token(word, rank_aliases, suit_aliases)
        if parsed_word is not None:
            return parsed_word

    rank = next((rank_aliases[word] for word in words if word in rank_aliases), None)
    suit = next((suit_aliases[word] for word in words if word in suit_aliases), None)
    if rank and suit:
        return rank, suit
    return None


def parse_compact_card_token(
    token: str,
    rank_aliases: dict[str, str],
    suit_aliases: dict[str, str],
) -> tuple[str, str] | None:
    normalized = str(token or "").strip().lower().replace("10", "t")
    match = re.fullmatch(r"([akqjt2-9])([shdc])", normalized)
    if match:
        rank_token, suit_token = match.group(1), match.group(2)
    else:
        match = re.fullmatch(r"([shdc])([akqjt2-9])", normalized)
        if not match:
            return None
        suit_token, rank_token = match.group(1), match.group(2)
    rank = rank_aliases.get("10" if rank_token == "t" else rank_token)
    suit = suit_aliases.get(suit_token)
    if rank and suit:
        return rank, suit
    return None


def extract_corner_glyphs(image: Any) -> tuple[Any, Any] | None:
    card = crop_likely_card(image)
    height, width = card.shape[:2]
    if height < 40 or width < 28:
        return None
    corner_w = max(20, int(width * 0.26))
    rank_h = max(24, int(height * 0.25))
    suit_y1 = max(0, int(height * 0.15))
    suit_y2 = min(height, max(suit_y1 + 20, int(height * 0.42)))
    rank_roi = card[0:rank_h, 0:corner_w]
    suit_roi = card[suit_y1:suit_y2, 0:corner_w]
    return normalize_bgr_piece(rank_roi, (54, 70)), normalize_bgr_piece(suit_roi, (42, 42))


def crop_likely_card(image: Any) -> Any:
    cv2, np = load_cv()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = ((hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 145)).astype("uint8") * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    contours, _hierarchy = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    height, width = image.shape[:2]
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.08:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = h / max(w, 1)
        score = area - abs(aspect - 1.4) * width * height * 0.03
        if best is None or score > best[0]:
            best = (score, x, y, w, h)
    if best is None:
        return image
    _score, x, y, w, h = best
    pad = max(1, int(min(w, h) * 0.02))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return image[y1:y2, x1:x2]


def normalize_bgr_piece(crop: Any, size: tuple[int, int]) -> Any:
    mask = foreground_mask(crop)
    return normalize_mask_piece(mask, size)


def foreground_mask(crop: Any) -> Any:
    cv2, _np = load_cv()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 45)
    dark = (hsv[:, :, 2] < 150) & (hsv[:, :, 1] < 205)
    return cv2.medianBlur((red | dark).astype("uint8") * 255, 3)


def normalize_mask_piece(mask: Any, size: tuple[int, int]) -> Any:
    cv2, np = load_cv()
    ys, xs = np.where(mask > 0)
    width, height = size
    if len(xs) == 0:
        return np.zeros((height, width), np.uint8)
    x1, x2 = max(0, int(xs.min()) - 1), min(mask.shape[1], int(xs.max()) + 2)
    y1, y2 = max(0, int(ys.min()) - 1), min(mask.shape[0], int(ys.max()) + 2)
    piece = mask[y1:y2, x1:x2]
    piece_h, piece_w = piece.shape
    scale = min(width / max(piece_w, 1), height / max(piece_h, 1))
    resized_w = max(1, int(piece_w * scale))
    resized_h = max(1, int(piece_h * scale))
    resized = cv2.resize(piece, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), np.uint8)
    x_offset = (width - resized_w) // 2
    y_offset = (height - resized_h) // 2
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    return canvas


def prepare_glyph_image(image: Any, kind: str) -> Any:
    cv2, np = load_cv()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    target = glyph_size(kind)
    if gray.shape != (target[1], target[0]):
        gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
    if gray.max() <= 1:
        gray = (gray * 255).astype(np.uint8)
    return gray.astype(np.uint8)


def make_feature_table(samples: list[tuple[str, Any, str]], kind: str, augment: int) -> tuple[Any, Any]:
    _cv2, np = load_cv()
    features = []
    labels = []
    for label, image, _source in samples:
        for variant in augment_glyph(prepare_glyph_image(image, kind), max(0, augment)):
            features.append(glyph_feature(variant, kind))
            labels.append(label)
    if not features:
        return np.zeros((0, feature_length(kind)), np.float32), np.array([], dtype="<U2")
    return np.stack(features).astype(np.float32), np.array(labels, dtype="<U2")


def merge_feature_tables(
    tables: list[tuple[Any, Any]],
    kind: str,
    *,
    seed_conflict_policy: str = "manual_override",
) -> tuple[Any, Any]:
    _cv2, np = load_cv()
    features = [features for features, _labels in tables if features.shape[0] > 0]
    labels = [labels for _features, labels in tables if labels.shape[0] > 0]
    if not features:
        return np.zeros((0, feature_length(kind)), np.float32), np.array([], dtype="<U2")
    seed_count = int(tables[0][0].shape[0]) if tables else 0
    merged_features = np.concatenate(features, axis=0).astype(np.float32)
    merged_labels = np.concatenate(labels, axis=0)
    return dedupe_feature_table(
        merged_features,
        merged_labels,
        kind,
        conflict_policy=seed_conflict_policy,
        protected_count=seed_count,
    )


def dedupe_feature_table(
    features: Any,
    labels: Any,
    kind: str,
    *,
    conflict_policy: str = "manual_override",
    protected_count: int = 0,
) -> tuple[Any, Any]:
    _cv2, np = load_cv()
    if conflict_policy not in {"manual_override", "keep_seed"}:
        raise ValueError("conflict_policy must be manual_override or keep_seed")
    if features.shape[0] == 0:
        return np.zeros((0, feature_length(kind)), np.float32), np.array([], dtype="<U2")
    by_hash: dict[bytes, tuple[Any, str, bool]] = {}
    order: list[bytes] = []
    for row_index, (feature, label) in enumerate(zip(features.astype(np.float32), labels.tolist())):
        key = feature.tobytes()
        if key not in by_hash:
            order.append(key)
        protected = row_index < max(0, int(protected_count))
        if conflict_policy == "keep_seed" and key in by_hash and by_hash[key][2]:
            continue
        # By default, later tables are more trusted. This lets manual labels
        # override an identical wrong prototype inherited from the current live
        # model. For diagnostic/baseline queues, keep_seed preserves an already
        # validated seed model when the queue only repeats existing prototypes.
        by_hash[key] = (feature, str(label), protected)
    deduped_features = [by_hash[key][0] for key in order]
    deduped_labels = [by_hash[key][1] for key in order]
    return np.stack(deduped_features).astype(np.float32), np.asarray(deduped_labels, dtype="<U2")


def augment_glyph(image: Any, augment: int) -> list[Any]:
    cv2, np = load_cv()
    variants = [image]
    if augment <= 0:
        return variants
    height, width = image.shape[:2]
    transforms = [
        (-2, 0, 1.0),
        (2, 0, 1.0),
        (0, -2, 1.0),
        (0, 2, 1.0),
        (-1, -1, 0.94),
        (1, 1, 1.06),
        (2, -1, 0.98),
        (-2, 1, 1.02),
        (0, 0, 0.90),
        (0, 0, 1.10),
    ]
    for dx, dy, scale in transforms[:augment]:
        matrix = np.array([[scale, 0, dx + width * (1 - scale) / 2], [0, scale, dy + height * (1 - scale) / 2]], dtype=np.float32)
        warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
        variants.append(warped)
    if augment >= 4:
        variants.append(cv2.GaussianBlur(image, (3, 3), 0))
    if augment >= 6:
        kernel = np.ones((2, 2), np.uint8)
        variants.append(cv2.dilate(image, kernel, iterations=1))
        variants.append(cv2.erode(image, kernel, iterations=1))
    return variants


def classify_rank_glyph(image: Any, model_path: Path | None = None) -> dict[str, Any] | None:
    return classify_glyph(image, "rank", model_path=model_path)


def classify_suit_glyph(
    image: Any,
    *,
    allowed: tuple[str, ...] | list[str] | None = None,
    model_path: Path | None = None,
) -> dict[str, Any] | None:
    return classify_glyph(image, "suit", model_path=model_path, allowed=allowed)


def classify_glyph(
    image: Any,
    kind: str,
    *,
    model_path: Path | None = None,
    allowed: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    model_path = resolve_card_classifier_model_path(model_path)
    model = load_card_classifier(model_path)
    if model is None:
        return None
    features = model[f"{kind}_features"]
    labels = model[f"{kind}_labels"]
    if features.shape[0] == 0:
        return None
    feature = glyph_feature(prepare_glyph_image(image, kind), kind)
    allowed_key = tuple(sorted(str(label) for label in allowed)) if allowed is not None else ()
    cache_key = (
        str(model_path.resolve()),
        kind,
        allowed_key,
        hashlib.blake2b(feature.tobytes(), digest_size=16).hexdigest(),
    )
    if cache_key in _GLYPH_CLASSIFY_CACHE:
        cached = _GLYPH_CLASSIFY_CACHE[cache_key]
        return dict(cached) if cached is not None else None
    metadata = model.get("metadata") or {}
    if bool(metadata.get("seed_guard")):
        seed_count = int(metadata.get(f"seed_{kind}_feature_count") or 0)
        if seed_count > 0:
            seed_result = classify_from_feature_table(
                features[: min(seed_count, int(features.shape[0]))],
                labels[: min(seed_count, int(labels.shape[0]))],
                feature,
                allowed=allowed,
            )
            if seed_result is not None and seed_result_passes_guard(seed_result, metadata, kind):
                guarded = {
                    **seed_result,
                    "model": str(model_path),
                    "seed_guard": True,
                }
                return store_glyph_classify_cache(cache_key, guarded)
    result = classify_from_feature_table(features, labels, feature, allowed=allowed)
    if result is None:
        store_glyph_classify_cache(cache_key, None)
        return None
    return store_glyph_classify_cache(cache_key, {**result, "model": str(model_path)})


def classify_from_feature_table(
    features: Any,
    labels: Any,
    feature: Any,
    *,
    allowed: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any] | None:
    if features.shape[0] == 0:
        return None
    scores = features @ feature
    if allowed is not None:
        allowed_set = set(allowed)
        keep = [idx for idx, label in enumerate(labels.tolist()) if label in allowed_set]
        if not keep:
            return None
        scores = scores[keep]
        labels = labels[keep]
    class_scores: dict[str, float] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if not bool(mask.any()):
            continue
        class_scores[label] = float(scores[mask].max())
    if not class_scores:
        return None
    ordered = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    label, score = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else -1.0
    return {
        "label": label,
        "score": float(score),
        "margin": float(score - second),
        "second_score": float(second),
    }


def seed_result_passes_guard(result: dict[str, Any], metadata: dict[str, Any], kind: str) -> bool:
    score = float(result.get("score") or 0.0)
    margin = float(result.get("margin") or 0.0)
    if kind == "rank":
        return (
            score >= float(metadata.get("seed_guard_rank_score") or 0.55)
            and margin >= float(metadata.get("seed_guard_rank_margin") or 0.10)
        )
    return (
        score >= float(metadata.get("seed_guard_suit_score") or 0.70)
        and margin >= float(metadata.get("seed_guard_suit_margin") or 0.04)
    )


def store_glyph_classify_cache(key: tuple[Any, ...], value: dict[str, Any] | None) -> dict[str, Any] | None:
    if len(_GLYPH_CLASSIFY_CACHE) >= _GLYPH_CLASSIFY_CACHE_LIMIT:
        _GLYPH_CLASSIFY_CACHE.clear()
    _GLYPH_CLASSIFY_CACHE[key] = dict(value) if value is not None else None
    return dict(value) if value is not None else None


def clear_glyph_classify_cache() -> None:
    _GLYPH_CLASSIFY_CACHE.clear()


def resolve_card_classifier_model_path(model_path: Path | None = None) -> Path:
    if model_path is not None:
        return Path(model_path)
    env_path = os.environ.get("GTO_CARD_KNN_MODEL")
    return Path(env_path) if env_path else DEFAULT_MODEL_PATH


def load_card_classifier(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    cv2, np = load_cv()
    del cv2
    resolved = model_path.resolve()
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]
    if not resolved.exists():
        _MODEL_CACHE[resolved] = None
        return None
    with np.load(str(resolved), allow_pickle=False) as data:
        model = {
            "rank_features": data["rank_features"].astype(np.float32),
            "rank_labels": data["rank_labels"].copy(),
            "suit_features": data["suit_features"].astype(np.float32),
            "suit_labels": data["suit_labels"].copy(),
            "metadata": json.loads(str(data["metadata"])),
        }
    _MODEL_CACHE[resolved] = model
    return model


def glyph_feature(image: Any, kind: str) -> Any:
    cv2, np = load_cv()
    image = prepare_glyph_image(image, kind)
    small = cv2.resize(image, feature_size(kind), interpolation=cv2.INTER_AREA)
    vector = small.astype(np.float32).reshape(-1) / 255.0
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def glyph_size(kind: str) -> tuple[int, int]:
    return (54, 70) if kind == "rank" else (42, 42)


def feature_size(kind: str) -> tuple[int, int]:
    return (32, 42) if kind == "rank" else (28, 28)


def feature_length(kind: str) -> int:
    width, height = feature_size(kind)
    return width * height


def rank_sort_key(label: str) -> int:
    try:
        return RANK_LABELS.index(label)
    except ValueError:
        return 999


def format_card_classifier_summary(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return f"train-card-classifier failed: {payload.get('error')}"
    metadata = payload["metadata"]
    lines = [
        f"Model: {payload['model']}",
        f"Rank samples: {metadata['rank_source_count']} source / {metadata['rank_feature_count']} augmented",
        f"Suit samples: {metadata['suit_source_count']} source / {metadata['suit_feature_count']} augmented",
        (
            "Sources: "
            f"seed rank/suit={metadata.get('seed_rank_feature_count', 0)}/{metadata.get('seed_suit_feature_count', 0)}, "
            f"seed_policy={metadata.get('seed_conflict_policy', 'manual_override')}, "
            f"template rank/suit={metadata.get('local_rank_source_count', 0)}/{metadata.get('local_suit_source_count', 0)}, "
            f"external rank/suit={metadata.get('external_rank_source_count', 0)}/{metadata.get('external_suit_source_count', 0)}, "
            f"glyph rank/suit={metadata.get('glyph_rank_source_count', 0)}/{metadata.get('glyph_suit_source_count', 0)}"
        ),
        f"Ranks: {', '.join(metadata['rank_labels'])}",
        f"Suits: {', '.join(metadata['suit_labels'])}",
        f"Skipped external images: {metadata['skipped_count']}",
    ]
    return "\n".join(lines)


def load_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required: pip install opencv-python numpy") from error
    return cv2, np
