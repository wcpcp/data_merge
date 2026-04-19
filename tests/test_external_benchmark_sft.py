from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_merge.external_benchmark_sft import (
    ERP_MULTIMODAL_SYSTEM_PROMPT,
    ExternalBenchmarkBuildConfig,
    build_external_benchmark_training_sets,
    build_panoenv_training_items_from_row,
    build_single_turn_training_item,
    build_thinking_record,
    sharegpt_user_text_to_blocks,
)


class ExternalBenchmarkSftTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="external_benchmark_sft_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_build_single_turn_training_item(self) -> None:
        image_path = self.temp_dir / "demo.jpg"
        image_path.write_bytes(b"demo")
        item = build_single_turn_training_item(
            record_id="osr_bench:demo:0",
            question="What object is visible?",
            answer="A chair.",
            image_paths=[image_path],
            system_prompt=ERP_MULTIMODAL_SYSTEM_PROMPT,
        )

        self.assertEqual(item["id"], "osr_bench:demo:0")
        self.assertEqual(item["messages"][0]["role"], "system")
        self.assertEqual(item["messages"][1]["role"], "user")
        self.assertEqual(item["messages"][1]["content"][0]["text"], "What object is visible?")
        self.assertEqual(item["messages"][1]["content"][1]["type"], "image")
        self.assertEqual(item["messages"][2]["content"][0]["text"], "A chair.")
        self.assertEqual(item["images"], [str(image_path)])

    def test_sharegpt_user_text_to_blocks_preserves_placeholder(self) -> None:
        blocks = sharegpt_user_text_to_blocks(
            "[initial observation]: <image>\nHuman Instruction: turn left",
            image_count=1,
            first_user_message=True,
        )
        self.assertEqual(blocks[0]["type"], "text")
        self.assertIn("[initial observation]: ", blocks[0]["text"])
        self.assertEqual(blocks[1]["type"], "image")
        self.assertEqual(blocks[2]["type"], "text")
        self.assertIn("turn left", blocks[2]["text"])

    def test_build_thinking_record_converts_sharegpt_shape(self) -> None:
        extract_dir = self.temp_dir / "thinking"
        image_path = extract_dir / "hos_sft_sharegpt" / "135_0_0.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"demo")

        item = build_thinking_record(
            row={
                "system": "Navigation system prompt",
                "images": ["hos_sft_sharegpt/135_0_0.png"],
                "conversations": [
                    {"from": "human", "value": "[initial observation]: <image>\nInstruction: move"},
                    {"from": "gpt", "value": "<think>plan</think><answer>move_forward</answer>"},
                ],
            },
            record_index=5,
            source_name="hos_sft",
            extract_dir=extract_dir,
        )

        assert item is not None
        self.assertEqual(item["id"], "thinking_in_360:hos_sft:000005")
        self.assertEqual(item["messages"][0]["content"][0]["text"], "Navigation system prompt")
        self.assertEqual(item["messages"][1]["role"], "user")
        self.assertEqual(item["messages"][1]["content"][1]["type"], "image")
        self.assertIn("move_forward", item["messages"][2]["content"][0]["text"])
        self.assertEqual(item["images"], [str(image_path)])

    def test_build_panoenv_training_items_from_row(self) -> None:
        image_path = self.temp_dir / "panoenv.jpg"
        image_path.write_bytes(b"demo")
        items = build_panoenv_training_items_from_row(
            {
                "env": "AbandonedCable",
                "image_id": "P000_001500",
                "questions": [
                    {
                        "question_id": 1,
                        "question": "What is the environment type?",
                        "answer": "abandoned cable site",
                    },
                    {
                        "question_id": 2,
                        "question": "How many doors are visible?",
                        "answer": 3,
                    },
                ],
            },
            image_path=image_path,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "panoenv:AbandonedCable:P000_001500:q1")
        self.assertEqual(items[0]["messages"][1]["content"][0]["text"], "What is the environment type?")
        self.assertEqual(items[1]["messages"][2]["content"][0]["text"], "3")
        self.assertEqual(items[0]["images"], [str(image_path)])

    def test_build_external_benchmark_training_sets_writes_empty_files_for_skipped_sources(self) -> None:
        output_dir = self.temp_dir / "output"
        cache_dir = self.temp_dir / "cache"
        stats = build_external_benchmark_training_sets(
            ExternalBenchmarkBuildConfig(
                output_dir=output_dir,
                cache_dir=cache_dir,
                include_osr_bench=False,
                include_thinking_in_360=False,
                include_panoenv=False,
            )
        )

        self.assertEqual(stats["counts"]["osr_bench"], 0)
        self.assertEqual(stats["counts"]["thinking_in_360"], 0)
        self.assertEqual(stats["counts"]["panoenv"], 0)
        self.assertIn("skipped_by_default", stats["notes"]["osr_bench"]["export_status"])
        self.assertIn("perspective_only", stats["notes"]["thinking_in_360"]["export_status"])
        self.assertEqual(json_load(output_dir / "osr_bench_training_multimodal_blocks.json"), [])
        self.assertEqual(json_load(output_dir / "thinking_in_360_training_multimodal_blocks.json"), [])
        self.assertEqual(json_load(output_dir / "panoenv_training_multimodal_blocks.json"), [])


def json_load(path: Path):
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
