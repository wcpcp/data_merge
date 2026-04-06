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

from data_merge.image_manifest import ImageManifestConfig, build_image_manifest, discover_image_paths, stable_stem


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
        self._write_test_image(image_root / f"{stem}.jpg", 64, 32)
        (dataset_root / "search_results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        payload = build_image_manifest(
            ImageManifestConfig(
                dataset_root=dataset_root,
                output_manifest_path=dataset_root / "image_manifest.json",
                dataset_name="real_360_test",
                workers=2,
                progress_every=1,
            )
        )

        self.assertEqual(payload["summary"]["image_count"], 1)
        self.assertEqual(payload["summary"]["metadata_match_count"], 1)
        image_record = payload["records"][0]
        self.assertEqual(image_record["image_path"], str(image_root / f"{stem}.jpg"))
        self.assertEqual(image_record["source"], "File:Demo Panorama.jpg")
        self.assertEqual(image_record["scene_id"], stem)
        self.assertEqual(image_record["viewpoint_id"], stem)

    def test_build_image_manifest_with_panox_metadata_jsonl(self) -> None:
        dataset_root = self.temp_dir / "panox"
        image_root = dataset_root / "images"
        image_root.mkdir(parents=True)

        image_name = "ARGENTINA__ec72b0a5-394b-438e-9ede-dae70761b17f.jpg"
        image_path = image_root / image_name
        self._write_test_image(image_path, 64, 32)

        row = {
            "index": "1",
            "id": "ec72b0a5-394b-438e-9ede-dae70761b17f",
            "collection": "3ba39738-8ec2-4329-9620-2462537143ba",
            "datetime": "2025-01-21T18:31:07+00:00",
            "asset_key": "hd",
            "asset_url": "https://panoramax.openstreetmap.fr/images/ec/72/b0/a5/394b-438e-9ede-dae70761b17f.jpg",
            "width": "5760",
            "height": "2880",
            "erp_reason": "gpano_equirectangular",
            "quality_grade": "B",
            "quality_score_value": "3.8000",
            "lon": "-67.3383761",
            "lat": "-43.727274",
            "local_path": str(image_path),
            "image_name": image_name,
            "source_metadata_csv": "/workspace/data_dir/data_user/ljh/Panorama/consolidated_output/ARGENTINA/metadata.csv",
            "source_region_dir": "/workspace/data_dir/data_user/ljh/Panorama/consolidated_output/ARGENTINA",
        }
        (dataset_root / "metadata.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        payload = build_image_manifest(
            ImageManifestConfig(
                dataset_root=dataset_root,
                output_manifest_path=dataset_root / "image_manifest.json",
                dataset_name="panox_test",
                workers=2,
                progress_every=1,
            )
        )

        self.assertEqual(payload["summary"]["image_count"], 1)
        self.assertEqual(payload["summary"]["metadata_match_count"], 1)
        image_record = payload["records"][0]
        self.assertEqual(image_record["image_path"], str(image_path))
        self.assertEqual(image_record["source"], row["asset_url"])
        self.assertEqual(image_record["scene_id"], Path(image_name).stem)
        self.assertEqual(image_record["viewpoint_id"], Path(image_name).stem)

    def test_discover_image_paths_streams_in_sorted_order(self) -> None:
        dataset_root = self.temp_dir / "streaming"
        image_root = dataset_root / "images"
        (image_root / "b").mkdir(parents=True)
        (image_root / "a").mkdir(parents=True)
        self._write_test_image(image_root / "b" / "02.jpg", 8, 4)
        self._write_test_image(image_root / "a" / "01.jpg", 8, 4)

        paths = [str(path.relative_to(image_root)) for path in discover_image_paths(image_root)]
        self.assertEqual(paths, ["a/01.jpg", "b/02.jpg"])

    def test_build_image_manifest_skips_broken_image(self) -> None:
        dataset_root = self.temp_dir / "broken"
        image_root = dataset_root / "images"
        image_root.mkdir(parents=True)

        good_path = image_root / "good.jpg"
        bad_path = image_root / "bad.jpg"
        self._write_test_image(good_path, 64, 32)
        bad_path.write_bytes(b"not-a-real-image")

        payload = build_image_manifest(
            ImageManifestConfig(
                dataset_root=dataset_root,
                output_manifest_path=dataset_root / "image_manifest.json",
                dataset_name="broken_test",
                workers=2,
                progress_every=1,
            )
        )

        self.assertEqual(payload["summary"]["image_count"], 1)
        self.assertEqual(payload["summary"]["error_count"], 1)
        self.assertEqual(len(payload["records"]), 1)
        self.assertEqual(payload["records"][0]["image_path"], str(good_path))
        self.assertEqual(len(payload["errors"]), 1)
        self.assertEqual(payload["errors"][0]["image_path"], str(bad_path))

    def _write_test_image(self, path: Path, width: int, height: int) -> None:
        try:
            from PIL import Image

            image = Image.new("RGB", (width, height), color=(120, 80, 40))
            image.save(path)
            return
        except Exception:
            pass

        import cv2
        import numpy as np

        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[:, :] = (40, 80, 120)
        if not cv2.imwrite(str(path), array):
            raise RuntimeError(f"failed to create test image: {path}")


if __name__ == "__main__":
    unittest.main()
