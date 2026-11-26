"""Pre-resize ISIC dataset images to 256x256 for fast CPU/GPU training."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from PIL import Image

from src.config import RAW_DATA_ROOT, RESIZED_DATA_ROOT


def _resize_one(args):
    src_path, dst_path, size = args
    if dst_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        img.save(dst_path, format="JPEG", quality=95)


def prepare_resized_dataset(size: int = 256, max_workers: int = 16) -> Path:
    t0 = time.time()
    tasks = []
    for split in ("Train", "Test"):
        split_dir = RAW_DATA_ROOT / split
        for img_path in split_dir.rglob("*.jpg"):
            rel = img_path.relative_to(RAW_DATA_ROOT)
            tasks.append((img_path, RESIZED_DATA_ROOT / rel, size))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_resize_one, tasks))

    print(f"Prepared {len(tasks)} resized ({size}x{size}) images in {time.time() - t0:.1f}s -> {RESIZED_DATA_ROOT}")
    return RESIZED_DATA_ROOT


if __name__ == "__main__":
    prepare_resized_dataset()
