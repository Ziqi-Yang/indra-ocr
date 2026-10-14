"""Grayscale test pages (.npy) for dump_golden.py: PDF pages rendered as the
pipeline renders them, and variants that exercise orientation (quarter and half
turns), deskew (small tilts) and the adaptive second detection pass (low dpi).

    uv run --no-project --with numpy --with opencv-python-headless --with pypdfium2 \
        python tools/make_inputs.py OUT_DIR PDF:PAGE[,PAGE...] ...
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium


def render(path, i, dpi):
    doc = pdfium.PdfDocument(path)
    arr = doc[i].render(scale=dpi / 72.0, grayscale=True).to_numpy()
    doc.close()
    return np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)


def tilt(g, deg):
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[2:]:
    path, pages = spec.rsplit(":", 1)
    stem = Path(path).stem.replace(".", "_")
    for p in pages.split(","):
        i = int(p)
        g = render(path, i, 200)
        np.save(out / f"{stem}_p{i}.npy", g)
        if i == int(pages.split(",")[0]):
            np.save(out / f"{stem}_p{i}_rot90.npy", np.ascontiguousarray(np.rot90(g)))
            np.save(out / f"{stem}_p{i}_rot180.npy", np.ascontiguousarray(np.rot90(g, 2)))
            np.save(out / f"{stem}_p{i}_tilt2.npy", tilt(g, 2.0))
            np.save(out / f"{stem}_p{i}_tiltm1.npy", tilt(g, -1.3))
            np.save(out / f"{stem}_p{i}_dpi90.npy", render(path, i, 90))
print(sorted(x.name for x in out.iterdir()))
