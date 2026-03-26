from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_merge.builder import BuildConfig, build_dataset
from data_merge.caption_parser import parse_caption_sections
from data_merge.normalizers import normalize_caption_record


class MergePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="data_merge_test_"))
        self.mock_root = ROOT / "examples" / "mock_input"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_build_dataset_with_mock_inputs(self) -> None:
        stats = build_dataset(
            BuildConfig(
                results_dir=self.mock_root / "results_final_v2",
                caption_jsonl=self.mock_root / "vqa_generation" / "output_v.jsonl",
                grounding_json=self.mock_root / "grounding" / "pano_grounding_train_factory.json",
                output_dir=self.temp_dir,
                workers=2,
            )
        )

        self.assertEqual(stats["results_final_v2_count"], 2)
        self.assertEqual(stats["caption_sft_count"], 7)
        self.assertEqual(stats["grounding_count"], 1)
        self.assertEqual(stats["merged_total_count"], 10)
        self.assertEqual(stats["training_data_count"], 10)
        self.assertEqual(stats["workers"], 2)

        merged_path = self.temp_dir / "merged_sft.jsonl"
        merged_rows = [json.loads(line) for line in merged_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(merged_rows), 10)

        results_rows = [row for row in merged_rows if row["source"] == "results_final_v2"]
        self.assertEqual(len(results_rows), 2)
        self.assertTrue(all(row["id"].startswith("scene_00001:") for row in results_rows))
        self.assertTrue(
            all(
                row["images"] == [
                    "/workspace/data_dir/data_user/public_data/360video/Realsee3D/real_world_data/scene_00001/viewpoints/1753781394/panoImage_1600.jpg"
                ]
                for row in results_rows
            )
        )

        caption_rows = [row for row in merged_rows if row["source"] == "caption_vqa"]
        self.assertTrue(any(row["subtask"] == "full_caption" for row in caption_rows))
        self.assertTrue(any(row["subtask"] == "global_layout" for row in caption_rows))

        for row in merged_rows:
            for image in row["images"]:
                self.assertNotIn("/workspace/data_dir/USB_data2", image)
                self.assertNotIn("/workspace/data_dir/USB_data", image)

        training_path = self.temp_dir / "training_data.json"
        training_rows = json.loads(training_path.read_text(encoding="utf-8"))
        self.assertEqual(len(training_rows), 10)
        self.assertEqual(set(training_rows[0].keys()), {"messages", "images"})
        self.assertTrue(training_rows[0]["messages"][0]["content"].startswith("<image>"))
        self.assertTrue(training_rows[-1]["messages"][0]["content"].startswith("<image>"))

    def test_parse_caption_sections(self) -> None:
        caption_path = self.mock_root / "vqa_generation" / "output_v.jsonl"
        record = json.loads(caption_path.read_text(encoding="utf-8").splitlines()[0])
        sections = parse_caption_sections(record["description"])
        self.assertIn("GLOBAL LAYOUT", sections)
        self.assertIn("FINAL RECONSTRUCTION", sections)
        self.assertIn("kitchen", sections["GLOBAL LAYOUT"].lower())

    def test_caption_record_with_null_description_does_not_crash(self) -> None:
        items = normalize_caption_record(
            record_index=0,
            record={
                "pano_path": "/workspace/data_dir/USB_data2/wcp_pano_training/vqa_generation/real_world_data/scene_x/viewpoints/1/panoImage_1600.jpg",
                "description": None,
                "mask_path": None,
            },
            include_full_caption=True,
            drop_missing_images=False,
        )
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
