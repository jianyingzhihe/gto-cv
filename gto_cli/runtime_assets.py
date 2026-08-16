from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARD_TEMPLATE_DIR = PROJECT_ROOT / "pict" / "card_templates"
REQUIRED_CARD_MODELS = (
    "card_glyph_knn.npz",
    "card_glyph_board_knn.npz",
    "card_glyph_hero_rank_knn.npz",
    "card_glyph_suit_knn.npz",
    "card_glyph_hero_black_suit_knn.npz",
    "card_glyph_board_black_suit_knn.npz",
)
REQUIRED_RANK_LABELS = frozenset("AKQJT98765432")
REQUIRED_SUIT_LABELS = frozenset("shdc")


def missing_runtime_card_assets(project_root: Path = PROJECT_ROOT) -> list[str]:
    root = Path(project_root)
    template_dir = root / "pict" / "card_templates"
    model_dir = root / "pict" / "card_models"
    missing = [
        str(model_dir / filename)
        for filename in REQUIRED_CARD_MODELS
        if not (model_dir / filename).is_file()
    ]
    rank_labels = {
        path.stem.removeprefix("rank_").split("_", 1)[0]
        for path in template_dir.glob("rank_*.png")
    }
    suit_labels = {
        path.stem.removeprefix("suit_").split("_", 1)[0]
        for path in template_dir.glob("suit_*.png")
    }
    for label in sorted(REQUIRED_RANK_LABELS - rank_labels):
        missing.append(f"{template_dir}\\rank_{label}_*.png")
    for label in sorted(REQUIRED_SUIT_LABELS - suit_labels):
        missing.append(f"{template_dir}\\suit_{label}_*.png")
    return missing


def require_runtime_card_assets(project_root: Path = PROJECT_ROOT) -> None:
    missing = missing_runtime_card_assets(project_root)
    if not missing:
        return
    preview = "\n".join(f"- {path}" for path in missing[:12])
    if len(missing) > 12:
        preview += f"\n- 另有 {len(missing) - 12} 项缺失"
    raise RuntimeError(
        "牌面识别运行资产不完整，当前电脑不能可靠识别手牌和公共牌：\n"
        f"{preview}\n"
        "请在项目目录执行 git pull，并运行 python -m pip install -r requirements.txt。"
    )
