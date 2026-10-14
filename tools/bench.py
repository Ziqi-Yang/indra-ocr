"""Throughput and latency of the Python package and the Rust CLI (both engines)
on the same inputs, runs interleaved so drift (thermal, other load) hits all
three alike.

    python tools/bench.py --pdfs DIR [--copies 6] [--cores 4,8,16,32] [--modes ocr,auto]
        [--rust target/release/indra-ocr] [--reps 2] [--out bench.json]

Needs INDRA_PY (the Python package's folder) and the environment both CLIs
need (models, ORT_DYLIB_PATH, PDFIUM_LIB). Throughput: all pages of ``copies``
copies of every PDF, wall time. Latency: the first 3 pages of each PDF in one
process (serial mode), the median of total_ms.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PY = os.environ.get("INDRA_PY", "/home/meowking/proj/probe/indra")


def cli(who: str, rust: str) -> list[str]:
    if who == "python":
        return [sys.executable, "-c", f"import sys; sys.path.insert(0, {PY!r}); from indra_ocr.cli import main; sys.exit(main())"]
    return [rust, "--engine", who]


def run(cmd: list[str]) -> tuple[float, str]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"{' '.join(cmd[:3])}...: exit {p.returncode}\n{p.stderr[-2000:]}")
    return time.perf_counter() - t0, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", type=Path, required=True)
    ap.add_argument("--copies", type=int, default=6)
    ap.add_argument("--cores", default="4,8")
    ap.add_argument("--modes", default="ocr,auto")
    ap.add_argument("--rust", default="target/release/indra-ocr")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--who", default="python,ort,native")
    ap.add_argument("--latency-cores", default="1,4")
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    who = a.who.split(",")
    pdfs = sorted(a.pdfs.glob("*.pdf"))
    tmp = Path(tempfile.mkdtemp(prefix="indra_bench_"))
    inputs = []
    for k in range(a.copies):
        for p in pdfs:
            q = tmp / f"{p.stem}_{k}.pdf"
            shutil.copy(p, q)
            inputs.append(str(q))
    # pages per copy set, from one JSON run
    _, out = run(cli("native", a.rust) + ["--format", "json", "--mode", "text", *map(str, pdfs)])
    pages = len(out.splitlines()) * a.copies
    results = {"pages": pages, "throughput": [], "latency": []}
    print(f"{len(inputs)} files, {pages} pages", flush=True)
    for mode in a.modes.split(","):
        for cores in map(int, a.cores.split(",")):
            for rep in range(a.reps):
                for w in who:
                    time.sleep(a.cooldown)
                    dt, _ = run(cli(w, a.rust) + ["--cores", str(cores), "--mode", mode, *inputs])
                    r = {"mode": mode, "cores": cores, "rep": rep, "who": w, "seconds": round(dt, 2), "pages_per_s": round(pages / dt, 2)}
                    results["throughput"].append(r)
                    print(json.dumps(r), flush=True)
        for cores in map(int, a.latency_cores.split(",")):
            for w in who:
                time.sleep(a.cooldown)
                ms = []
                for p in pdfs:
                    _, out = run(cli(w, a.rust) + ["--cores", str(cores), "--mode", mode, "--max-pages", "3", "--format", "json", str(p)])
                    ms += [json.loads(l)["timings"]["total_ms"] for l in out.splitlines() if l.strip()]
                r = {"mode": mode, "cores": cores, "who": w, "median_ms": round(statistics.median(ms), 1), "pages": len(ms)}
                results["latency"].append(r)
                print(json.dumps(r), flush=True)
    shutil.rmtree(tmp)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
