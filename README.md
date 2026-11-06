# U-DHRL

Source code for *An Unsupervised Method for Anomaly Detection in Business Processes Based on Deep Hierarchical Reinforcement Learning*.

The repository includes raw event-log preprocessing, anomaly injection, binary detection, baseline comparisons, and on-demand inference.

## Install

Python 3.10 is recommended. Create an isolated environment and install:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, install the matching PyTorch CUDA build. See [dependencies](docs/DEPENDENCIES.md) and [third-party code](docs/THIRD_PARTY.md).

## Binary detection

Provide local XES or XES.GZ paths using `logs.example.json` as a template. Datasets are supplied separately.

```bash
python pipeline.py --logs logs.local.json --output runs/main --device cuda:0
```

The default method is U-DHRL. Use `--models U-DHRL Naive DAE BINet TranAD GRASPED` for the full comparison after preparing the optional GRASPED dependency.

## Checks

```bash
python smoke_test.py --output runs/synthetic_check
```

The smoke check trains the binary detector on synthetic data with a reduced budget. See [validation checks](docs/VALIDATION.md).

## Layout

- `pipeline.py`: raw logs through binary training and evaluation.
- `src/`: representations, evidence, policies, baselines and evaluation modules.
- `vendor/`: required third-party helper and its license.
- `docs/`: data, dependencies, architecture and validation notes.

Dataset sources are listed in [DATASETS.md](docs/DATASETS.md). Training writes model weights and experiment outputs to the selected run directory.
