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

A laptop under sustained load throttles; numbers are the mean of 2 interleaved runs.

Throughput, 288 pages, 8 cores (pages/s):

| mode | python | rust ort | rust native |
|---|---|---|---|
| ocr | 8.1 | 7.9 | **17.1** (2.1x) |
| auto | 28.5 | 36.9 | **45.7** (1.6x) |

Latency, one page at a time (median ms per page):

| mode | cores | python | rust ort | rust native |
|---|---|---|---|---|
| ocr | 1 | 625 | 634 | **378** |
| ocr | 4 | 281 | 282 | **158** |
| auto | 1 | 108 | 106 | **101** |
| auto | 4 | 87 | 81 | **81** |

One scanned page (`assets/examples/multi_column.png`, 138 lines, OCR): python 1027 / 416 / 985 ms on
1 / 4 / 16 cores, rust native 541 / 210 / 162 ms (ONNX Runtime slows down with more threads than
physical cores; the native thread pool keeps scaling).

Where a page's time goes (one core, OCR mode, a paper page): recognizer 210 ms native vs 438 ms on
ONNX Runtime (the products run at ~140 GFLOP/s, near this core's fp32 peak), detector 31 vs 62 ms,
orientation model 10 vs 17 ms.

## Intel Xeon Silver 4314 (Ice Lake SP, 32 vCPUs, VM)

_Pending: tools/bench.py run in progress._
