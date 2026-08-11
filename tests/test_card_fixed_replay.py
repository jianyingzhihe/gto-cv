from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from gto_cli.card_fixed_replay import (
    infer_fixed_board_boxes,
    load_existing_glyph_labels,
    migrate_existing_glyph_labels,
    select_fixed_hero_details,
)
from gto_cli.video_vision import load_cv
from gto_cli.card_glyph_label_queue import GLYPH_LABEL_QUEUE_COLUMNS


class CardFixedReplayTest(unittest.TestCase):
    def test_shifted_h2_falls_back_to_complete_raw_h2(self) -> None:
        shifted = [{"index": 1, "card": "??", "roi_mode": "shifted"}]
        raw = [{"index": 1, "card": "4d", "roi_mode": "raw"}]

        selected = select_fixed_hero_details(shifted, raw)

        self.assertEqual(selected[0]["card"], "4d")
        self.assertEqual(selected[0]["roi_mode"], "raw")

    def test_complete_shifted_h2_wins_for_overlapped_ten(self) -> None:
        shifted = [{"index": 1, "card": "Td", "rank_confidence": 0.91, "roi_mode": "shifted"}]
        raw = [{"index": 1, "card": "8d", "rank_confidence": 0.72, "roi_mode": "raw"}]

        selected = select_fixed_hero_details(shifted, raw)

        self.assertEqual(selected[0]["card"], "Td")
        self.assertEqual(selected[0]["roi_mode"], "shifted")

    def test_complete_raw_seven_beats_complete_shifted_black_king_hint(self) -> None:
        shifted = [{"index": 1, "card": "Ks", "rank_confidence": 0.6815, "roi_mode": "shifted"}]
        raw = [{"index": 1, "card": "7c", "rank_confidence": 0.8055, "roi_mode": "raw"}]

        selected = select_fixed_hero_details(shifted, raw)

        self.assertEqual(selected[0]["card"], "7c")
        self.assertEqual(selected[0]["roi_mode"], "raw")

    def test_decisive_clean_raw_h2_beats_complete_shifted_misread(self) -> None:
        shifted = [{"index": 1, "card": "Jh", "rank_source": "", "roi_mode": "shifted"}]
        raw = [{"index": 1, "card": "5h", "rank_source": "clean_corner", "roi_mode": "raw"}]

        selected = select_fixed_hero_details(shifted, raw)

        self.assertEqual(selected[0]["card"], "5h")
        self.assertEqual(selected[0]["roi_mode"], "raw")

    def test_incomplete_raw_seven_beats_shifted_black_king_hint(self) -> None:
        shifted = [
            {
                "index": 1,
                "card": "K?",
                "rank_confidence": 0.6815,
                "rank_margin": 0.3631,
                "roi_mode": "shifted",
            }
        ]
        raw = [
            {
                "index": 1,
                "card": "7?",
                "rank_confidence": 0.8055,
                "rank_margin": 0.1020,
                "roi_mode": "raw",
            }
        ]

        selected = select_fixed_hero_details(shifted, raw)

        self.assertEqual(selected[0]["card"], "7?")
        self.assertEqual(selected[0]["roi_mode"], "raw")

    def test_infers_stable_board_boxes_from_slot_medians(self) -> None:
        records = [
            {
                "saved": [
                    {
                        "group": "board",
                        "slot": slot,
                        "roi_box": {
                            "x": 296 + slot * 100 + offset,
                            "y": 352 + offset,
                            "width": 100,
                            "height": 137,
                        },
                    }
                    for slot in range(5)
                ]
            }
            for offset in (-1, 0, 1)
        ]

        boxes = infer_fixed_board_boxes(records, frame_width=1100, frame_height=777)

        self.assertEqual(len(boxes), 5)
        self.assertAlmostEqual(boxes[0]["x"], 296 / 1100)
        self.assertAlmostEqual(boxes[0]["y"], 352 / 777)
        self.assertAlmostEqual(boxes[4]["x"], 696 / 1100)
        self.assertAlmostEqual(boxes[4]["width"], 100 / 1100)
        self.assertAlmostEqual(boxes[4]["height"], 137 / 777)

    def test_migrates_label_by_physical_slot_not_old_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_sample = root / "old" / "samples" / "sample_0001"
            new_sample = root / "new" / "samples" / "sample_0001"
            old_sample.mkdir(parents=True)
            new_sample.mkdir(parents=True)
            old_input = old_sample / "hero_slot0_Kc_rank.png"
            new_input = new_sample / "hero_slot0_Ts_rank.png"
            old_input.write_bytes(b"old")
            new_input.write_bytes(b"new")
            old_queue = root / "old_queue.csv"
            new_queue = root / "new_queue.csv"
            self._write_queue(
                old_queue,
                [
                    {
                        "label_id": "G0001",
                        "kind": "rank",
                        "input_path": str(old_input),
                        "current_label": "K",
                        "final_label": "T",
                        "notes": "migrated_from_old_crop",
                    }
                ],
            )
            self._write_queue(
                new_queue,
                [
                    {
                        "label_id": "G0042",
                        "kind": "rank",
                        "input_path": str(new_input),
                        "current_label": "T",
                    }
                ],
            )

            migrated = migrate_existing_glyph_labels(old_queue_csv=old_queue, new_queue_csv=new_queue)

            with new_queue.open("r", encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(migrated, 1)
            self.assertEqual(row["final_label"], "T")
            self.assertIn("migrated_from_old_crop", row["notes"])
            self.assertEqual(row["notes"].count("migrated_from_old_crop"), 1)

    def test_preserved_current_queue_label_overrides_older_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "samples" / "sample_0001"
            sample.mkdir(parents=True)
            source = sample / "hero_slot1_Jd_rank.png"
            source.write_bytes(b"x")
            old_queue = root / "old.csv"
            current_queue = root / "current.csv"
            new_queue = root / "new.csv"
            base = {
                "label_id": "G0001",
                "kind": "rank",
                "input_path": str(source),
                "current_label": "J",
            }
            self._write_queue(old_queue, [{**base, "final_label": "J"}])
            self._write_queue(current_queue, [{**base, "final_label": "K"}])
            self._write_queue(new_queue, [base])

            migrated = migrate_existing_glyph_labels(
                old_queue_csv=old_queue,
                new_queue_csv=new_queue,
                preserved_labels=load_existing_glyph_labels(current_queue),
            )

            with new_queue.open("r", encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(migrated, 1)
            self.assertEqual(row["final_label"], "K")

    def test_migrates_identical_suit_glyph_from_another_replay_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_input = root / "old" / "board_slot0_7c_suit.png"
            new_input = root / "new" / "hero_slot1_7c_suit.png"
            old_input.parent.mkdir(parents=True)
            new_input.parent.mkdir(parents=True)
            cv2, np = load_cv()
            glyph = np.zeros((18, 12), dtype=np.uint8)
            glyph[3:15, 4:8] = 255
            cv2.imwrite(str(old_input), glyph)
            cv2.imwrite(str(new_input), glyph)
            old_queue = root / "old.csv"
            new_queue = root / "new.csv"
            self._write_queue(
                old_queue,
                [{"label_id": "G0001", "kind": "suit", "input_path": str(old_input), "final_label": "c"}],
            )
            self._write_queue(
                new_queue,
                [{"label_id": "G0042", "kind": "suit", "input_path": str(new_input), "current_label": "?"}],
            )

            migrated = migrate_existing_glyph_labels(old_queue_csv=old_queue, new_queue_csv=new_queue)

            with new_queue.open("r", encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(migrated, 1)
            self.assertEqual(row["final_label"], "c")
            self.assertIn("migrated_from_identical_glyph", row["notes"])

    @staticmethod
    def _write_queue(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=GLYPH_LABEL_QUEUE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
