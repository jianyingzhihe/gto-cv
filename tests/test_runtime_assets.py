from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gto_cli.runtime_assets import (
    REQUIRED_CARD_MODELS,
    missing_runtime_card_assets,
    require_runtime_card_assets,
)


class RuntimeAssetsTest(unittest.TestCase):
    def test_complete_runtime_asset_set_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_dir = root / "pict" / "card_templates"
            model_dir = root / "pict" / "card_models"
            template_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            for label in "AKQJT98765432":
                (template_dir / f"rank_{label}.png").touch()
            for label in "shdc":
                (template_dir / f"suit_{label}.png").touch()
            for filename in REQUIRED_CARD_MODELS:
                (model_dir / filename).touch()

            self.assertEqual(missing_runtime_card_assets(root), [])
            require_runtime_card_assets(root)

    def test_missing_assets_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaisesRegex(RuntimeError, "git pull"):
                require_runtime_card_assets(root)
