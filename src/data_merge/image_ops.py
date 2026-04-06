from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import shutil
import subprocess
import warnings


_IMAGE_BACKEND: Optional[str] = None


def configure_pillow_for_large_images() -> None:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)


def resize_image_in_place(path: Path, width: int, height: int) -> None:
    backend = detect_image_backend()
    if backend == "pillow":
        resize_with_pillow(path, width, height)
        return
    if backend == "opencv":
        resize_with_opencv(path, width, height)
        return
    if backend == "ffmpeg":
        resize_with_ffmpeg(path, width, height)
        return
    raise RuntimeError("No supported image backend found. Install Pillow, opencv-python, or ffmpeg.")


def detect_image_backend() -> Optional[str]:
    global _IMAGE_BACKEND
    if _IMAGE_BACKEND is not None:
        return _IMAGE_BACKEND
    try:
        from PIL import Image  # noqa: F401

        configure_pillow_for_large_images()
        _IMAGE_BACKEND = "pillow"
        return _IMAGE_BACKEND
    except Exception:
        pass
    try:
        import cv2  # noqa: F401

        _IMAGE_BACKEND = "opencv"
        return _IMAGE_BACKEND
    except Exception:
        pass
    if shutil.which("ffmpeg"):
        _IMAGE_BACKEND = "ffmpeg"
        return _IMAGE_BACKEND
    return None


def resize_with_pillow(path: Path, width: int, height: int) -> None:
    from PIL import Image

    configure_pillow_for_large_images()
    with Image.open(path) as image:
        if image.size == (width, height):
            return
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        save_kwargs = {}
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": 95})
        resized.save(path, **save_kwargs)


def resize_with_opencv(path: Path, width: int, height: int) -> None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read image for resize: {path}")
    current_height, current_width = image.shape[:2]
    if (current_width, current_height) == (width, height):
        return
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), resized):
        raise RuntimeError(f"failed to write resized image: {path}")


def resize_with_ffmpeg(path: Path, width: int, height: int) -> None:
    temp_path = path.with_name(f"{path.stem}__resized{path.suffix}")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vf",
                f"scale={width}:{height}:flags=lanczos",
                str(temp_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    backend = detect_image_backend()
    if backend == "pillow":
        from PIL import Image

        configure_pillow_for_large_images()
        with Image.open(path) as image:
            return image.size
    if backend == "opencv":
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height
    return None
