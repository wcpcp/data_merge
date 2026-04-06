from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_merge.image_ops import configure_pillow_for_large_images


class ImageOpsTest(unittest.TestCase):
    def test_configure_pillow_for_large_images(self) -> None:
        try:
            from PIL import Image, ImageFile
        except Exception:
            self.skipTest("Pillow not installed")

        previous_max_pixels = Image.MAX_IMAGE_PIXELS
        previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            Image.MAX_IMAGE_PIXELS = 123456
            ImageFile.LOAD_TRUNCATED_IMAGES = False

            configure_pillow_for_large_images()

            self.assertIsNone(Image.MAX_IMAGE_PIXELS)
            self.assertTrue(ImageFile.LOAD_TRUNCATED_IMAGES)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_max_pixels
            ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated


if __name__ == "__main__":
    unittest.main()
