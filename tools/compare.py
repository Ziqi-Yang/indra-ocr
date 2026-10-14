"""Run the Python and the Rust CLIs on the same inputs and compare their JSON
output page by page (everything but timings).

    uv run --no-project --with numpy --with opencv-python-headless --with onnxruntime \
        --with pypdfium2 --with huggingface_hub python tools/compare.py \
        [--rust target/release/indra-ocr] [-- CLI ARGS] INPUTS...

Exit code 1 when any page differs.
"""

import argparse
import json
import os
import subprocess
import sys
import time

PY = os.environ.get("INDRA_PY", "/home/meowking/proj/probe/indra")


def run(cmd, env=None):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.perf_counter() - t0
    pages = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    return pages, p.stdout, p.stderr, p.returncode, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", default="target/release/indra-ocr")
    ap.add_argument("--args", default="", help="extra CLI arguments for both, e.g. '--mode ocr --math'")
    ap.add_argument("--show", type=int, default=3, help="differences to print per field")
    ap.add_argument("inputs", nargs="+")
    a = ap.parse_args()
    extra = a.args.split()
    py_cmd = [sys.executable, "-c", f"import sys; sys.path.insert(0, {PY!r}); from indra_ocr.cli import main; sys.exit(main())"]
    py, py_raw, py_err, py_rc, py_t = run(py_cmd + ["--format", "json", *extra, *a.inputs])
    rs, rs_raw, rs_err, rs_rc, rs_t = run([a.rust, "--format", "json", *extra, *a.inputs])
    print(f"python: {len(py)} pages in {py_t:.2f}s (exit {py_rc}); rust: {len(rs)} pages in {rs_t:.2f}s (exit {rs_rc})")
    if py_rc != rs_rc:
        print("exit codes differ\n--- python stderr\n" + py_err[-2000:] + "\n--- rust stderr\n" + rs_err[-2000:])
    bad = 0
    key = lambda p: (p["file"], p["page"])
    pyd, rsd = {key(p): p for p in py}, {key(p): p for p in rs}
    for k in sorted(set(pyd) | set(rsd)):
        if k not in pyd or k not in rsd:
            print(f"{k}: only in {'rust' if k in rsd else 'python'}")
            bad += 1
            continue
        p, r = pyd[k], rsd[k]
        diffs = []
        for f in ("source", "rotation", "skew", "note", "markdown", "blocks"):
            if p[f] != r[f]:
                diffs.append(f)
        pl, rl = p["lines"], r["lines"]
        if len(pl) != len(rl):
            diffs.append(f"lines ({len(pl)} vs {len(rl)})")
        else:
            nd = [i for i, (x, y) in enumerate(zip(pl, rl)) if x["text"] != y["text"] or x["box"] != y["box"] or abs(x["conf"] - y["conf"]) > 2e-4]
            if nd:
                diffs.append(f"lines {nd[:10]}")
        if diffs:
            bad += 1
            print(f"{k}: {', '.join(diffs)}")
            if "note" in diffs:
                print(f"   note: {p['note']!r} vs {r['note']!r}")
            if "markdown" in diffs and p["markdown"] and r["markdown"]:
                for x, y in zip(p["markdown"].splitlines(), r["markdown"].splitlines()):
                    if x != y:
                        print(f"   python: {x[:160]}\n   rust:   {y[:160]}")
                        break
    # byte-level check of the shared fields' formatting
    if not bad:
        strip = lambda raw: "\n".join(json.dumps({k: v for k, v in json.loads(l).items() if k != "timings"}, ensure_ascii=False) for l in raw.splitlines() if l.strip())
        same_bytes = all(l1.split(', "timings"')[0] == l2.split(', "timings"')[0] for l1, l2 in zip(py_raw.splitlines(), rs_raw.splitlines()))
        print(f"all {len(py)} pages match" + ("" if same_bytes else " (JSON formatting differs)"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
