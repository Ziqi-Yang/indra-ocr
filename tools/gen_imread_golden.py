"""Image files in the formats Indra OCR reads, and cv2.imread(..., GRAYSCALE)
(cv2.imreadmulti for TIFF) of each, for crates/indra-ocr/tests/imread_golden.rs.

    uv run --no-project --with numpy --with opencv-python-headless --with pillow \
        python tools/gen_imread_golden.py tests/fixtures/golden/imread
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/golden/imread")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
h, w = 97, 131
rgb = rng.integers(0, 256, (h, w, 3), np.uint8)
rgb[10:40, 10:60] = (200, 200, 200)  # grey patch: r == g == b
rgba = np.dstack([rgb, rng.integers(0, 256, (h, w), np.uint8)])
gray = rng.integers(0, 256, (h, w), np.uint8)
files = {}
Image.fromarray(rgb).save(out / "rgb.png"); files["rgb.png"] = 1
Image.fromarray(rgba).save(out / "rgba.png"); files["rgba.png"] = 1
Image.fromarray(gray).save(out / "gray.png"); files["gray.png"] = 1
Image.fromarray(rgb).quantize(64).save(out / "palette.png"); files["palette.png"] = 1
cv2.imwrite(str(out / "rgb16.png"), (rgb.astype(np.uint16) * 257 + rng.integers(0, 200, rgb.shape).astype(np.uint16))); files["rgb16.png"] = 1
Image.fromarray(gray).convert("1").save(out / "bilevel.png"); files["bilevel.png"] = 1
cv2.imwrite(str(out / "color.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90]); files["color.jpg"] = 1
cv2.imwrite(str(out / "gray.jpg"), gray); files["gray.jpg"] = 1
Image.fromarray(rgb).save(out / "sub420.jpg", quality=85, subsampling=2); files["sub420.jpg"] = 1
cv2.imwrite(str(out / "rgb.bmp"), rgb[..., ::-1]); files["rgb.bmp"] = 1
cv2.imwrite(str(out / "rgb.webp"), rgb[..., ::-1], [cv2.IMWRITE_WEBP_QUALITY, 101]); files["rgb.webp"] = 1
pages = [Image.fromarray(gray), Image.fromarray(rgb), Image.fromarray(gray[::-1])]
pages[0].save(out / "multi.tif", save_all=True, append_images=pages[1:]); files["multi.tif"] = 3
Image.fromarray(gray).convert("1").save(out / "g4.tif", compression="group4"); files["g4.tif"] = 1
for f in files:
    ok, frames = cv2.imreadmulti(str(out / f), flags=cv2.IMREAD_GRAYSCALE)
    if not ok or not frames:
        frames = [cv2.imread(str(out / f), cv2.IMREAD_GRAYSCALE)]
    for i, fr in enumerate(frames):
        np.save(out / f"{f}.{i}.npy", fr)
    files[f] = len(frames)
(out / "manifest.json").write_text(json.dumps(files, indent=1))
print(files)
