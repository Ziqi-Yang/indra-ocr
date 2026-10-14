"""Golden outputs of the Python Indra OCR pipeline, stage by stage, for the
Rust port's parity tests.

    INDRA_PY=/path/to/indra uv run --no-project --with numpy --with opencv-python-headless \
        --with onnxruntime --with pypdfium2 --with huggingface_hub \
        python tools/dump_golden.py OUT_DIR IMAGE_OR_NPY...

For each input (an image file, or a .npy grayscale page) OUT_DIR/<name>/ gets:
  gray.npy        the page as Python decoded it (uint8)
  upright.npy     the page after orientation and deskew
  stages.json     orientation, skew, detector boxes, recognizer output
  page.json       the final Page (as `indra-ocr --format json` writes it)
Options: --math (math models on), --threads N.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("INDRA_PY", "/home/meowking/proj/probe/indra"))

import cv2  # noqa: E402
import indra_ocr  # noqa: E402
from indra_ocr.pipeline import read_images  # noqa: E402


def page_json(page) -> dict:
    return {
        "source": page.source,
        "rotation": page.rotation,
        "skew": page.skew,
        "note": page.note,
        "lines": [asdict(ln) for ln in page.lines],
        "markdown": page.markdown,
        "blocks": page.blocks,
        "timings": page.timings,
    }


def tok(t):
    return [t[0] if isinstance(t[0], str) else int(t[0]), float(t[1])] + [int(v) for v in t[2:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--math", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    conv = indra_ocr.load(a.model_dir, threads=a.threads, mode="ocr", math=a.math)
    ocr = conv.ocr
    for p in a.inputs:
        frames = [np.load(p)] if p.suffix == ".npy" else read_images(p)
        for fi, gray in enumerate(frames):
            name = p.stem + (f"_{fi}" if len(frames) > 1 else "") + ("_math" if a.math else "")
            d = a.out / name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "gray.npy", gray)
            up, k, skew = ocr.upright(gray)
            np.save(d / "upright.npy", up)
            boxes = ocr.detect(up)
            rec = ocr.recognize(up, boxes)
            page = ocr(gray)
            stages = {
                "orientation": int(k),
                "skew": float(skew),
                "boxes": boxes.tolist(),
                "rec": [[t, float(c), [tok(x) for x in toks]] for t, c, toks in rec],
            }
            (d / "stages.json").write_text(json.dumps(stages, ensure_ascii=False))
            (d / "page.json").write_text(json.dumps(page_json(page), ensure_ascii=False, indent=1))
            print(f"{name}: {len(page.lines)} lines, rotation {page.rotation}, skew {page.skew}")


if __name__ == "__main__":
    main()
