# U-DHRL: standalone source release

This package runs the core U-DHRL binary-detection experiment from raw public event logs. It reconstructs case partitions and anomaly recipes, trains representation and policy networks from scratch, evaluates frozen models and writes summary tables. No old server directories, split files, checkpoints, prediction caches or results are required.

## Installation

Use Python 3.10–3.12. Linux, Windows and WSL are supported by the standalone entry point. Install a hardware-appropriate PyTorch 2.3.1 build and the requirements:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA 12.1, install the corresponding PyTorch build from the official PyTorch wheel index. CPU execution is available but complete experiments may be slow. The NumPy/scikit-learn pins specify the release environment; their exact historical experiment versions were not recoverable and are not claimed to be identical.

## Data

Download the public XES/XES.GZ logs listed in DATASETS.md. Copy logs.example.json to logs.local.json and set the raw-log file paths. Windows JSON paths can use forward slashes. Only raw logs and installed software are required.

## Complete pipeline

```bash
python pipeline.py --logs logs.local.json --output runs/main --device cuda:0
```

Default: U-DHRL, five seeds (42, 123, 456, 789, 2024), every dataset supplied in the mapping, primary injection rates 5–45% and supplementary rates 1%, 2%. Use `--device cpu` when CUDA is unavailable. Outputs must be new directories; completed results are never silently overwritten.

For a small functional check:

```bash
python pipeline.py --logs logs.local.json --output runs/smoke --seeds 42 --models U-DHRL Naive --device cpu --smoke
```

`--smoke` reduces training epochs/candidates. Its scores are not publication results. For all baselines:

```bash
python setup_grasped.py
python pipeline.py --logs logs.local.json --output runs/six_methods --models U-DHRL Naive DAE BINet GRASPED TranAD --device cuda:0
```

GRASPED is optional. setup_grasped.py fetches its upstream implementation at the recorded revision; no author-held files are needed. See THIRD_PARTY.md. The U-DHRL-only flow does not need GRASPED.

For preprocessing/injection only, add `--prepare-only`. Data are split before injection; training, validation and calibration contain no injected anomalies. Split IDs and realized operations are recorded in each run. Thresholds/model selection are frozen before test evaluation.

## Outputs

- `data/`: generated arrays, training-derived vocabulary, case partitions and injection recipes.
- `development/`: freshly trained weights, validation selection and thresholds.
- `results/`: predictions and per-rate binary metrics.
- `summary/all_metrics.csv`: individual dataset/seed/rate results.
- `summary/binary_summary.json`: equal-rate seed means, then mean/sample SD across seeds. One-seed SD is null.
- `RUN_COMPLETE.json`: successful completion and whether smoke settings were used.

All outputs are local runtime artifacts and are not included in this release.

## On-demand inference

After a complete main run:

```bash
python benchmark.py --run runs/main --dataset BPIC13_C --seed 42 --device cuda:0
```

This reuses the models just trained by this pipeline and compares eager, on-demand and all-fine execution. It checks binary/class agreement between eager and on-demand outputs. The helper measures full-test-pass latency; it does not reproduce the paper's dedicated single-case and 128-case timing protocol.

## Scope and correspondence to the manuscript

The core VAE, calibrated evidence, rewards, low-level action sets and policy training modules are retained from the supplied main-experiment source. The standalone wrapper replaces historical orchestration and generates new deterministic splits and recipes using the archive perturbation constructors and current event-level replay. Historical V23/V25 data and audit files are not loaded.

No claim of bitwise agreement with the original paper results is made without running the full experiment on the same raw log versions and numerical environment. Raw-log SHA256, splits, recipes and configuration are written for each new run. A successful smoke test validates execution, not the reported performance.

The main experiment compares U-DHRL with five baselines on case-level binary detection. All methods use the same evaluation rule: normal cases form the negative class, and cases with an injected control-flow anomaly form the positive class.

SOURCE_MANIFEST.json records the provenance of the core files. src/standalone_data.py is new integration code calling only archive helpers. Historical recipe equality is not claimed. Some retained utility functions refer to historical paths but are not called by pipeline.py; the standalone entry point constructs all inputs it uses. See VALIDATION.md for checks actually performed.

Synthetic end-to-end check: `python smoke_test.py --output runs/synthetic_check`. This generates its own test log; no dataset download is needed for the functional check.
