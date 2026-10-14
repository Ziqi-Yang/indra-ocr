"""Golden outputs of the OpenCV calls Indra OCR makes, for the Rust ports in
crates/indra-ocr/src/imgproc.rs.

    uv run --no-project --with numpy --with opencv-python-headless \
        python tools/gen_cv_golden.py tests/fixtures/golden/cv

Each case is a folder of .npy arrays plus an entry in manifest.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def page_like(rng, h, w):
    """A page-like uint8 image: white paper with dark strokes and some noise."""
    img = np.full((h, w), 245, np.uint8)
    for _ in range(max(1, h * w // 400)):
        y, x = rng.integers(0, h), rng.integers(0, w)
        hh, ww = rng.integers(1, max(2, h // 8)), rng.integers(1, max(2, w // 4))
        img[y : y + hh, x : x + ww] = rng.integers(0, 120)
    noise = rng.integers(-20, 20, (h, w))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cases = []

    def save(name, kind, params, **arrays):
        d = out / name
        d.mkdir(exist_ok=True)
        for k, v in arrays.items():
            np.save(d / f"{k}.npy", np.ascontiguousarray(v))
        cases.append({"name": name, "kind": kind, **params})

    # resize, uint8: the interpolations and size pairs the pipeline uses
    shapes = [(37, 53), (100, 80), (64, 64), (33, 401), (1200, 900), (17, 5), (3, 2)]
    k = 0
    for h, w in shapes:
        src = page_like(rng, h, w) if h * w > 50 else rng.integers(0, 256, (h, w), np.uint8)
        targets = {
            (max(1, w // 2), max(1, h // 2)),
            (max(1, w // 3), max(1, h // 3)),
            (max(1, round(w * 0.37)), max(1, round(h * 0.37))),
            (max(1, round(w * 1.7)), max(1, round(h * 1.7))),
            (max(8, round(w * 32 / h)), 32),
            (w + 1, h),
            (w, max(1, h - 1)),
        }
        for tw, th in sorted(targets):
            for interp, iname in ((cv2.INTER_LINEAR, "linear"), (cv2.INTER_AREA, "area")):
                dst = cv2.resize(src, (tw, th), interpolation=interp)
                save(f"resize_u8_{k}", "resize_u8", {"interp": iname}, src=src, dst=dst)
                k += 1
    # resize, float32 linear (line mask probabilities)
    for i, (h, w, tw, th) in enumerate([(24, 80, 230, 70), (96, 400, 1301, 311), (10, 10, 3, 4)]):
        src = rng.random((h, w), np.float32)
        dst = cv2.resize(src, (tw, th), interpolation=cv2.INTER_LINEAR)
        save(f"resize_f32_{i}", "resize_f32", {"interp": "linear"}, src=src, dst=dst)

    # warpAffine: rotate_page (uint8, replicate border) and the skew search
    # (float32 ink, constant 0 border)
    for i, (h, w, a) in enumerate([(300, 220, 1.3), (301, 223, -4.7), (120, 800, 0.3), (64, 64, 45.0)]):
        src = page_like(rng, h, w)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        dst = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        save(f"warp_u8_{i}", "warp_u8", {"angle": a}, src=src, dst=dst, M=M)
        ink = (255.0 - src.astype(np.float32)) / 255.0
        size = (w, h)
        c = (size[0] / 2, size[1] / 2)
        M2 = cv2.getRotationMatrix2D(c, a, 1.0)
        dst2 = cv2.warpAffine(ink, M2, size)
        save(f"warp_f32_{i}", "warp_f32", {"angle": a}, src=ink, dst=dst2, M=M2)

    # connected components, 4-connectivity, on detector-like masks
    for i, (h, w) in enumerate([(64, 48), (512, 384), (7, 300)]):
        m = (rng.random((h, w)) > 0.55).astype(np.uint8)
        m[h // 3 : h // 3 + 3, 2 : w - 2] = 1
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        save(f"cc_{i}", "cc", {"n": int(n)}, src=m, labels=lab.astype(np.int32), stats=stats.astype(np.int32))

    (out / "manifest.json").write_text(json.dumps({"opencv": cv2.__version__, "cases": cases}, indent=1))
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/golden/cv"))
