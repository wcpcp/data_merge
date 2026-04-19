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
    assign_group_to_split,
    build_external_benchmark_training_sets,
    build_panorama_angle_prompt,
    build_panoenv_training_items_from_row,
    build_single_turn_training_item,
    build_thinking_panorama_target_records,
    build_thinking_records,
    build_thinking_record,
    build_thinking_rl_records,
    choose_best_rl_initial_view,
    sharegpt_user_text_to_blocks,
    strip_reasoning_tags,
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

    def test_build_thinking_records_from_panorama_sft_json(self) -> None:
        extract_dir = self.temp_dir / "thinking_pano"
        manifest_dir = extract_dir / "hos_sft_panorama" / "135"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        image_path = manifest_dir / "frame_0001.jpg"
        image_path.write_bytes(b"demo")
        manifest_path = manifest_dir / "sft.json"
        manifest_path.write_text("[]", encoding="utf-8")

        items = build_thinking_records(
            row={
                "task": "Find the supermarket exit",
                "outputs": [
                    {
                        "input_angles": [180, 0],
                        "content": "<think>Look toward the door.</think><action>rotate(45,0)</action>",
                        "action": "rotate(45,0)",
                    }
                ],
            },
            record_index=7,
            source_name="hos_sft_panorama",
            extract_dir=extract_dir,
            manifest_path=manifest_path,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "thinking_in_360:hos_sft_panorama:000007:step000")
        self.assertIn("yaw=180", items[0]["messages"][1]["content"][0]["text"])
        self.assertIn("rotate(45,0)", items[0]["messages"][2]["content"][0]["text"])
        self.assertEqual(items[0]["images"], [str(image_path)])

    def test_build_thinking_panorama_target_records_uses_final_submit_only(self) -> None:
        extract_dir = self.temp_dir / "thinking_pano_target"
        manifest_dir = extract_dir / "hos_sft_panorama" / "135"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        image_path = manifest_dir / "frame_0001.jpg"
        image_path.write_bytes(b"demo")
        manifest_path = manifest_dir / "sft.json"
        manifest_path.write_text("[]", encoding="utf-8")

        items = build_thinking_panorama_target_records(
            row={
                "task": "look for a store selling sunglasses",
                "outputs": [
                    {
                        "input_angles": [0, 0],
                        "content": "<think>rotate</think><action>rotate(8,6)</action>",
                        "action": "rotate(8,6)",
                    },
                    {
                        "input_angles": [8, 6],
                        "content": "<think>submit</think><action>submit(8,6)</action>",
                        "action": "submit(8,6)",
                    },
                ],
            },
            record_index=0,
            source_name="hos_sft_panorama",
            extract_dir=extract_dir,
            manifest_path=manifest_path,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "thinking_in_360:hos_sft_panorama:000000")
        self.assertEqual(items[0]["images"], [str(image_path)])
        self.assertIn("image center is yaw=0", items[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(items[0]["messages"][2]["content"][0]["text"], "[8.00, 6.00]")

    def test_build_thinking_records_can_strip_reasoning(self) -> None:
        extract_dir = self.temp_dir / "thinking_strip"
        manifest_dir = extract_dir / "hos_sft_panorama" / "9"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        image_path = manifest_dir / "frame_0002.jpg"
        image_path.write_bytes(b"demo")
        manifest_path = manifest_dir / "sft.json"
        manifest_path.write_text("[]", encoding="utf-8")

        items = build_thinking_records(
            row={
                "task": "Locate the red box",
                "outputs": [
                    {
                        "input_angles": [90, 10],
                        "content": "<think>First inspect the shelves.</think><action>rotate(-30,0)</action>",
                    }
                ],
            },
            record_index=1,
            source_name="hos_sft_panorama",
            extract_dir=extract_dir,
            manifest_path=manifest_path,
            strip_reasoning=True,
        )

        self.assertEqual(items[0]["messages"][2]["content"][0]["text"], "rotate(-30,0)")

    def test_choose_best_rl_initial_view_prefers_overlap_then_level(self) -> None:
        index = choose_best_rl_initial_view(
            initial_yaws=[0, 90, 180, 270],
            overlaps=[0.1, 0.7, 0.7, 0.2],
            levels=[3, 2, 1, 0],
        )
        self.assertEqual(index, 2)

    def test_build_thinking_rl_records_prefers_panorama_and_best_yaw(self) -> None:
        extract_dir = self.temp_dir / "thinking_rl"
        sample_dir = extract_dir / "hos_train_rl" / "307"
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_dir / "pano_023.png"
        image_path.write_bytes(b"demo")
        manifest_path = sample_dir / "annotation.json"
        manifest_path.write_text("[]", encoding="utf-8")

        items = build_thinking_rl_records(
            row={
                "task": "Find the goods with $10 off discount",
                "yaw": [56.5, 87.8],
                "pitch": [-34.0, 6.9],
            },
            record_index=0,
            source_name="hos_train_rl",
            extract_dir=extract_dir,
            manifest_path=manifest_path,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["images"], [str(image_path)])
        self.assertEqual(items[0]["messages"][2]["content"][0]["text"], "[72.15, -13.55]")

    def test_strip_reasoning_tags_keeps_action_payload(self) -> None:
        self.assertEqual(
            strip_reasoning_tags("<think>reasoning</think><action>move_forward</action>"),
            "move_forward",
        )

    def test_build_panorama_angle_prompt_uses_erp_convention(self) -> None:
        prompt = build_panorama_angle_prompt("Locate the target")
        self.assertIn("yaw=0", prompt)
        self.assertIn("-180", prompt)
        self.assertIn("180", prompt)

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
                        "question_type": "open_ended",
                        "question": "What is the environment type?",
                        "answer": "abandoned cable site",
                    },
                    {
                        "question_id": 2,
                        "question_type": "multiple_choice",
                        "question": "How many doors are visible?",
                        "answer": 3,
                    },
                ],
            },
            image_path=image_path,
            allowed_question_types={"open_ended"},
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "panoenv:AbandonedCable:P000_001500:q1")
        self.assertEqual(items[0]["messages"][1]["content"][0]["text"], "What is the environment type?")
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

        self.assertEqual(stats["counts"]["osr_bench"]["total"], 0)
        self.assertEqual(stats["counts"]["thinking_in_360"], 0)
        self.assertEqual(stats["counts"]["panoenv"], 0)
        self.assertIn("skipped_by_default", stats["notes"]["osr_bench"]["export_status"])
        self.assertEqual(stats["notes"]["thinking_in_360"]["export_status"], "skipped_by_request")
        self.assertEqual(json_load(output_dir / "osr_bench_train_multimodal_blocks.json"), [])
        self.assertEqual(json_load(output_dir / "osr_bench_validation_multimodal_blocks.json"), [])
        self.assertEqual(json_load(output_dir / "osr_bench_test_multimodal_blocks.json"), [])
        self.assertEqual(json_load(output_dir / "thinking_in_360_training_multimodal_blocks.json"), [])
        self.assertEqual(json_load(output_dir / "panoenv_training_multimodal_blocks.json"), [])

    def test_assign_group_to_split_is_deterministic(self) -> None:
        split_a = assign_group_to_split("image_a")
        split_b = assign_group_to_split("image_a")
        self.assertEqual(split_a, split_b)
        self.assertIn(split_a, {"train", "validation", "test"})


def json_load(path: Path):
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
