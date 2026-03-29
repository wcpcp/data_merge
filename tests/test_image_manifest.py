from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_merge.image_manifest import ImageManifestConfig, build_image_manifest, stable_stem


class ImageManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="image_manifest_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_build_image_manifest_with_metadata_match(self) -> None:
        dataset_root = self.temp_dir / "real_360"
        image_root = dataset_root / "images"
        image_root.mkdir(parents=True)

        row = {
            "source": "commons",
            "source_id": "File:Demo Panorama.jpg",
            "provider": "wikimedia_commons",
            "title": "Demo Panorama",
            "caption": "A demo ERP image.",
            "width": 4096,
            "height": 2048,
            "is_360": True,
        }
        stem = stable_stem(row)
        (image_root / f"{stem}.jpg").write_bytes(b"fake-image")
        (dataset_root / "search_results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        payload = build_image_manifest(
            ImageManifestConfig(
                dataset_root=dataset_root,
                output_manifest_path=dataset_root / "image_manifest.json",
                dataset_name="real_360_test",
                workers=2,
            )
        )

        self.assertEqual(payload["summary"]["image_count"], 1)
        self.assertEqual(payload["summary"]["metadata_match_count"], 1)
        image_record = payload["images"][0]
        self.assertEqual(image_record["dataset"], "real_360_test")
        self.assertEqual(image_record["source"], "commons")
        self.assertEqual(image_record["title"], "Demo Panorama")
        self.assertEqual(image_record["width"], 4096)
        self.assertEqual(image_record["height"], 2048)
        self.assertTrue(image_record["metadata_match_found"])


if __name__ == "__main__":
    unittest.main()
