# indra-ocr (Rust)

A Rust port of [Indra OCR](https://github.com/datalab-to/indra) (Datalab's fast CPU OCR: line detection,
CTC recognition, reading order, layout and markdown, for PDFs and images), with its own inference engine.

- **Same output as the Python package.** On the test documents, every page's lines, boxes, markdown and
  blocks match `indra-ocr` 0.3.1 in every mode (auto, ocr, text, math). With `--engine ort` the JSON is
  byte-for-byte identical; the native engine differs only in the 4th decimal of some confidences.
- **Faster.** The native engine (`indra-engine`) runs every model with fused, CPU-dispatched kernels:
  about 2x the Python package's throughput on scanned pages, and 2-6x lower single-page latency.
- **No Python, no PyTorch, no ONNX Runtime needed.** The native engine reads the weights straight from
  the ONNX files. PDFs need the pdfium shared library; ONNX Runtime is optional (`--engine ort`).

## Crates

| crate | what |
|---|---|
| `indra-ocr` | the pipeline: decoding, orientation and deskew, detection post-processing, line crops, CTC decoding, line splitting, reading order, layout, markdown, the PDF text layer, `convert_many` |
| `indra-engine` | the native inference engine: an ONNX weight reader, packed GEMM micro-kernels (AVX-512, AVX2/FMA, NEON, portable), and the models (recognizer, detector, orientation, line classifiers, order transformer, math encoder/decoder) |
| `indra-ocr-cli` | the `indra-ocr` and `indra-ocr-viewer` binaries |

## Build

```shell
nix develop          # rust, pdfium, onnxruntime (sets PDFIUM_DYNAMIC_LIB_PATH, ORT_DYLIB_PATH)
cargo build --release
```

Without Nix: any Rust >= 1.88, plus a pdfium build for PDFs (for example
[pdfium-binaries](https://github.com/bblanchon/pdfium-binaries), or the `libpdfium.so` inside the
`pypdfium2_raw` Python wheel). ONNX Runtime is only needed for `--engine ort`.

Cargo features of `indra-ocr`: `native` (the native engine), `ort` (ONNX Runtime), `pdf` (pdfium),
`download` (models from Hugging Face). All are on by default; `--no-default-features --features native,pdf`
builds without ONNX Runtime and without network code.

## Use

```shell
indra-ocr paper.pdf                       # markdown to stdout
indra-ocr *.pdf --out results/            # one .md per input, all cores
indra-ocr scan.png --format json          # lines, boxes, layout blocks, confidences, timings
indra-ocr paper.pdf --mode ocr --timings  # ignore the text layer; per-page timings on stderr
indra-ocr paper.pdf --math                # math as LaTeX
indra-ocr paper.pdf --engine ort          # the models on ONNX Runtime instead
indra-ocr-viewer                          # local web viewer on http://localhost:8765
```

The options are the Python CLI's (`--mode`, `--model-order`, `--no-image-ocr`, `--cores`, `--dpi`,
`--max-pages`, `--password`, `--math`, `--keep-furniture`, `--timings`, `--model-dir`), plus `--engine`.

From Rust:

```rust
let opts = indra_ocr::LoadOptions::default();          // native engine, models from the HF cache
let mut conv = indra_ocr::load(&opts, 4)?;             // 4 threads
for (i, page) in conv.convert("paper.pdf".as_ref(), 0, None)? {
    println!("page {i}: {}", page.markdown.unwrap_or_default());
}
```

`indra_ocr::convert_many` reads many files on all cores (one single-threaded pipeline per worker thread,
one page per task) and hands back each file's pages in input order.

### Environment

| variable | |
|---|---|
| `INDRA_OCR_MODEL_DIR` | a local model folder (otherwise the Hugging Face cache, then a download of `datalab-to/indra-ocr` v3.1) |
| `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE` | as for huggingface_hub (the cache layout is shared with Python) |
| `PDFIUM_LIB`, `PDFIUM_DYNAMIC_LIB_PATH` | the pdfium library file, or its folder |
| `ORT_DYLIB_PATH` | the ONNX Runtime library (for `--engine ort`) |
| `INDRA_FORCE_ISA` | `avx512`, `avx2`, `neon` or `scalar`: use narrower kernels than the CPU allows |
| `INDRA_OCR_PDFIUM_COPIES=0` | share one pdfium between workers (see below) |

## How it matches the Python package

The pipeline's thresholds were tuned on the exact numbers Python produces, so the port reproduces them:

- **OpenCV**: `resize` (fixed-point INTER_LINEAR, INTER_AREA), `warpAffine` (OpenCV 4.11+'s float/FMA
  kernel), `connectedComponentsWithStats` and `getRotationMatrix2D` are ported and bit-exact against
  OpenCV 5.0; `imread(GRAYSCALE)` follows each codec's conversion (libpng's `rgb_to_gray`, imgcodecs,
  `cvtColor`) and is bit-exact except JPEG (±1 grey level on ~2% of pixels from the IDCT).
- **Python and NumPy semantics**: `str.isalpha`/`isdigit`/`isspace`/... and `unicodedata` from tables
  generated with Python (`tools/gen_unicode_tables.py`), `round` (half to even), `np.median`,
  `np.percentile`, NumPy's pairwise float32 summation, `np.arange`, stable-order ties.
- **pdfium**: the same render call as pypdfium2 and the same text-layer reads.
- **The models**: each native model is tested against ONNX Runtime (`tests/native_vs_ort.rs`): the
  recognizer's classes agree on all 442,620 timesteps of the test pages, detector boxes within 0.01 px.
  Where ONNX Runtime's result depends on batching (the line classifiers and the math encoder see the
  zero padding of their batch), the native engine reproduces the batch's padded width; where ONNX
  Runtime deviates from the ONNX spec (pooling output sizes), it follows ONNX Runtime.

Tests: `cargo test` (unit tests; the golden tests skip without fixtures). Golden data comes from the
Python package: `tools/gen_cv_golden.py`, `tools/gen_imread_golden.py`, `tools/dump_golden.py`;
`tools/compare.py` runs both CLIs on the same inputs and diffs their JSON page by page.

## The native engine

- **Weights, not graphs**: each model is a known architecture. `indra-engine` reads the ONNX
  initializers and checks the graph is the expected one (an unknown model revision is an error at load
  time, not a wrong answer). BatchNorm is folded into the preceding conv at load.
- **GEMM**: weights are packed once into K x NR panels (NR = 32/16/8 by matrix width); MR x NR
  micro-kernels broadcast A and stream B, with bias, activation (HardSwish, ReLU) and residual add fused
  into the store. About 140 GFLOP/s on one Zen 4 core.
- **Recognizer**: the ONNX graph pads a batch and masks every block; the native recognizer runs each line
  at its own length instead (no padding work), with lines packed into chunks of ~384 timesteps so the
  products stay in L2. The squeeze-excite, CTC argmax and softmax confidence are fused per chunk (the
  confidence only where the CTC path keeps a class).
- **CNNs** (detector, orientation): a compiled NHWC executor with conv+activation(+residual) fusion,
  depthwise kernels specialised per channel count, and the graph's shape arithmetic evaluated on
  integers.
- **Transformers** (reading order, math): small fused attention over the page's lines; the math decoder
  keeps per-line KV caches.
- **ISA dispatch** at run time (`is_x86_feature_detected!` / NEON), so one binary is fast on any x86-64
  or AArch64 CPU.
- **Threads**: page-parallel workers run single-threaded models; a single page (serial mode) splits its
  recognizer chunks, conv rows and line work over the cores. With the native engine the default worker
  count is one per physical core (SMT siblings only add contention to kernels that already keep the
  vector units busy); `--cores` overrides it.

### pdfium and threads

pdfium is not thread-safe, and its state is per loaded library. Each parallel worker therefore loads a
private copy of the pdfium library file (copied to a temporary folder and removed once loaded), so pages
render in parallel within one process, as the Python package's worker processes do.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md).

## License

The code is Apache 2.0 (see [LICENSE](LICENSE)); it ports the Apache 2.0 Python package
[datalab-to/indra](https://github.com/datalab-to/indra) (commit 661fb47), and the viewer serves that
package's page. The model weights are not part of this repository: they are Datalab's, under a modified
OpenRAIL-M license ([LICENSE](https://huggingface.co/datalab-to/indra-ocr/blob/main/LICENSE)), which
applies to any use of them through this code too.
