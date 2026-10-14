# Benchmarks

Three implementations on the same inputs, runs interleaved (`tools/bench.py`):

- **python**: the Python package `indra-ocr` 0.3.1 (ONNX Runtime 1.31, OpenCV 5.0, pypdfium2 5.14)
- **rust ort**: this port with `--engine ort` (the same ONNX Runtime 1.31 library)
- **rust native**: this port with the native engine (the default)

Inputs: two arXiv papers (48 pages), 6 copies of each for throughput (288 pages). Throughput: all pages,
one single-threaded pipeline per core (both CLIs' default), wall time including model loading.
Latency: the first 3 pages of each paper in one process with all the given cores on each page, median
`total_ms`. "ocr" ignores the PDF text layer (every page is detected and recognized); "auto" reads the
text layer where it passes the checks (most of these pages).

## AMD Ryzen 9 7940HS (Zen 4, 8 cores / 16 threads, laptop)

Mean of 2 interleaved runs (a laptop under sustained load throttles; runs agree within ~3%).

Throughput, 288 pages, 8 cores (pages/s):

| mode | python | rust ort | rust native |
|---|---|---|---|
| ocr | 8.1 | 7.9 | **19.1** (2.4x) |
| auto | 32.7 | 38.6 | **63.4** (1.9x) |

Latency, one page at a time (median ms per page):

| mode | cores | python | rust ort | rust native |
|---|---|---|---|---|
| ocr | 1 | 606 | 626 | **345** |
| ocr | 4 | 287 | 276 | **128** |
| auto | 1 | 110 | 105 | **84** |
| auto | 4 | 85 | 78 | **62** |

Where a scanned page's time goes (one core): recognizer ~210 ms native vs ~440 ms on ONNX Runtime (its
products run at ~140 GFLOP/s, near this core's fp32 peak), detector ~28 vs ~60 ms, orientation model
~8 vs ~16 ms. `--int8` takes the recognizer down another ~25%.

## Intel Xeon Silver 4314 (Ice Lake SP, 32 vCPUs, shared VM)

A VM on a shared host (other users' load during the runs; memory bandwidth ~9 GB/s for the whole VM),
AVX-512 with one FMA unit per core. Mean of 2 interleaved runs.

Throughput, 288 pages (pages/s):

| mode | cores | python | rust ort | rust native |
|---|---|---|---|---|
| ocr | 4 | 5.1 | 5.2 | **7.9** (1.5x) |
| ocr | 8 | 9.2 | 9.2 | **15.4** (1.7x) |
| ocr | 16 | 14.5 | 14.3 | **28.2** (1.9x) |
| ocr | 32 | 16.8 | 15.7 | **42.5** (2.5x) |
| auto | 4 | 15.8 | 24.3 | **29.2** (1.8x) |
| auto | 8 | 22.7 | 40.2 | **51.9** (2.3x) |
| auto | 16 | 46.3 | 54.3 | **82.3** (1.8x) |
| auto | 32 | 58.3 | 50.5 | **104.5** (1.8x) |

Latency, one page at a time (median ms per page):

| mode | cores | python | rust ort | rust native |
|---|---|---|---|---|
| ocr | 1 | 738 | 792 | **515** |
| ocr | 4 | 385 | 379 | **301** |
| auto | 1 | 191 | 155 | **145** |
| auto | 4 | 156 | **119** | 138 |

On this machine the native detector only matches ONNX Runtime's (single thread and on 4 threads), so
text-layer pages on 4 threads, which are mostly detection, are a little slower than with `--engine ort`;
everything with recognition is faster.

Raw results: `bench/bench-xeon-4314.json`, `bench/bench-ryzen-7940hs.json`.
