# Technical Text Ranking

Code for the paper "Separating Topic Supervision, Hierarchical Scoring and Local Mixing in Technical Text Ranking".

## Setup

Python 3.11, all commands from the repo root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows: `py -3.11 -m venv .venv`, then `.venv\Scripts\Activate.ps1` in PowerShell.

The paper's runs are CPU-only. The optional BGE runner also runs on Apple MPS. <!-- TODO: CUDA 是否可用/是否实测 -->

## Try it without the data

```bash
python -m unittest discover -s tests -v
python scripts/run_demo.py
```

The demo builds synthetic Chinese technical texts, a topic tree and a graph, then pushes one small component-disjoint experiment through all 13 local scoring methods. Two sanity checks ride along: random ranking and character deletion. Everything goes into `demo_output/`, and none of the patent data is touched. If that directory already exists, point `--output-dir` at another empty path.

The scores it prints are demo output. They will not line up with any table in the paper, and they are not meant to.

## Rebuilding the tables and figures

Tthis part needs no raw data:

```bash
python scripts/write_revision_tex.py
python scripts/make_revision_figures.py
```

Figures are written to `figure/revision/` as SVG, PDF and PNG, with input hashes recorded in `source_manifest.json`. Figures 1–6 belong to the revised experiments. Figures 7 and 8 are diagnostics on the older split and are not revised results.

## Full experiments

```bash
python scripts/run_revision_experiments.py --seeds 17 29 43
```

Four CSVs go in `data/`: patents, document topics, the topic tree, and graph edges. A fifth file, `retrieval_pairs.csv`, is only used by the old-split diagnostics. `--prepare-only` builds splits and pools without fitting anything, which is a quick way to check the inputs parse.

A full run writes new files into `results/revision/` and overwrites same-named summaries. If you want to keep the bundled snapshot around for comparison, run it in a fresh checkout.

