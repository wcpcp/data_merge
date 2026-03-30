from __future__ import annotations

import unittest

from pathlib import Path
import tempfile
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_merge.video_frames import (
    build_error_rows,
    build_failed_record,
    build_frame_manifest_rows,
    build_tail_timestamp_attempts,
    cleanup_partial_outputs,
    compute_extraction_frame_indices,
    compute_extraction_timestamps,
    compute_safe_tail_timestamp,
    compute_uniform_frame_indices,
    compute_uniform_timestamps,
    expected_output_paths,
)


class VideoFramesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_frames_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_compute_uniform_timestamps(self) -> None:
        timestamps = compute_uniform_timestamps(10.0, 5)
        self.assertEqual([round(value, 3) for value in timestamps], [1.0, 3.0, 5.0, 7.0, 9.0])

    def test_compute_extraction_timestamps_uses_video_tail_for_last_frame(self) -> None:
        timestamps = compute_extraction_timestamps(10.0, 5)
        self.assertEqual([round(value, 3) for value in timestamps[:-1]], [1.0, 3.0, 5.0, 7.0])
        self.assertAlmostEqual(timestamps[-1], 8.5, places=3)

    def test_compute_uniform_frame_indices(self) -> None:
        indices = compute_uniform_frame_indices(100, 5)
        self.assertEqual(indices, [10, 30, 50, 70, 90])

    def test_compute_extraction_frame_indices_uses_last_video_frame(self) -> None:
        indices = compute_extraction_frame_indices(100, 5)
        self.assertEqual(indices, [10, 30, 50, 70, 84])

    def test_expected_output_paths(self) -> None:
        paths = expected_output_paths(Path("/tmp/demo"), 3, ".jpg")
        self.assertEqual(
            [str(path) for path in paths],
            ["/tmp/demo/frame_00.jpg", "/tmp/demo/frame_01.jpg", "/tmp/demo/frame_02.jpg"],
        )

    def test_build_tail_timestamp_attempts_moves_back_from_video_end(self) -> None:
        attempts = build_tail_timestamp_attempts(10.0)
        self.assertAlmostEqual(attempts[0], 8.5, places=3)
        self.assertIn(7.5, [round(value, 3) for value in attempts])

    def test_compute_safe_tail_timestamp(self) -> None:
        self.assertAlmostEqual(compute_safe_tail_timestamp(10.0), 8.5, places=3)

    def test_build_frame_manifest_rows_uses_short_ids(self) -> None:
        rows = build_frame_manifest_rows(
            [
                {
                    "source_video_path": "/workspace/data_dir/data_user/public_data/360video/Sphere360/videos/-076WPWoCRE_137.0_152.0.mp4",
                    "frames": [
                        {"image_path": "/workspace/data_dir/data_user/public_data/360video/outdoor/images/Sphere360/-076WPWoCRE_137.0_152.0/frame_00.jpg"}
                    ],
                }
            ]
        )
        self.assertEqual(
            rows,
            [
                {
                    "image_path": "/workspace/data_dir/data_user/public_data/360video/outdoor/images/Sphere360/-076WPWoCRE_137.0_152.0/frame_00.jpg",
                    "source": "/workspace/data_dir/data_user/public_data/360video/Sphere360/videos/-076WPWoCRE_137.0_152.0.mp4",
                    "scene_id": "-076WPWoCRE_137.0_152.0",
                    "viewpoint_id": "frame_00",
                }
            ],
        )

    def test_cleanup_partial_outputs(self) -> None:
        paths = expected_output_paths(self.temp_dir, 5, ".jpg")
        for path in paths[:4]:
            path.write_bytes(b"x")
        cleanup_partial_outputs(paths)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_build_error_rows(self) -> None:
        rows = build_error_rows(
            [
                {
                    "dataset": "Sphere360",
                    "source_video_path": "/workspace/demo/a.mp4",
                    "relative_video_path": "a.mp4",
                    "status": "partial_error",
                    "frame_count_extracted": 4,
                    "error": "ffprobe failed",
                },
                {
                    "dataset": "panowan",
                    "source_video_path": "/workspace/demo/b.mp4",
                    "relative_video_path": "b.mp4",
                    "status": "ok",
                    "error": "",
                },
            ]
        )
        self.assertEqual(
            rows,
            [
                {
                    "dataset": "Sphere360",
                    "source_video_path": "/workspace/demo/a.mp4",
                    "relative_video_path": "a.mp4",
                    "status": "partial_error",
                    "frame_count_extracted": "4",
                    "error": "ffprobe failed",
                }
            ],
        )

    def test_build_failed_record_preserves_partial_frames(self) -> None:
        image_paths = expected_output_paths(self.temp_dir, 5, ".jpg")[:3]
        for path in image_paths:
            path.write_bytes(b"x")
        record = build_failed_record(
            dataset_name="Sphere360",
            source_root=Path("/workspace/source"),
            video_path=Path("/workspace/source/demo.mp4"),
            output_dir=self.temp_dir,
            image_paths=image_paths,
            frames_per_video=5,
            error="failed to extract tail frame",
        )
        self.assertEqual(record["status"], "partial_error")
        self.assertEqual(record["frame_count_extracted"], 3)
        self.assertEqual(len(record["frames"]), 3)


if __name__ == "__main__":
    unittest.main()
