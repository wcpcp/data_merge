from __future__ import annotations

import unittest

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_merge.video_frames import compute_uniform_frame_indices, compute_uniform_timestamps, expected_output_paths


class VideoFramesTest(unittest.TestCase):
    def test_compute_uniform_timestamps(self) -> None:
        timestamps = compute_uniform_timestamps(10.0, 5)
        self.assertEqual([round(value, 3) for value in timestamps], [1.0, 3.0, 5.0, 7.0, 9.0])

    def test_compute_uniform_frame_indices(self) -> None:
        indices = compute_uniform_frame_indices(100, 5)
        self.assertEqual(indices, [10, 30, 50, 70, 90])

    def test_expected_output_paths(self) -> None:
        paths = expected_output_paths(Path("/tmp/demo"), 3, ".jpg")
        self.assertEqual(
            [str(path) for path in paths],
            ["/tmp/demo/frame_00.jpg", "/tmp/demo/frame_01.jpg", "/tmp/demo/frame_02.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
