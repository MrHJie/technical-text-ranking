# Technical Text Ranking

Source code for **Separating Topic Supervision, Hierarchical Scoring and Local Mixing in Technical Text Ranking**.

[Data format](data/README.md)

## Install

Use Python 3.11. Run commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell. The experiments use CPU; the optional BGE runner supports CPU and Apple MPS.

## Run without the original dataset

```bash
python -m unittest discover -s tests -v
python scripts/run_demo.py
```

The demo constructs independent **synthetic** Chinese technical texts, a topic tree and a graph, then runs one small component-disjoint experiment through all 13 local scoring methods. It also runs random-ranking and character-deletion checks. It uses a separate `demo_output/` directory and no original patent data. Its scores are demonstration outputs, not paper results. Use `--output-dir PATH` for a different empty output directory.

Rebuild the paper's tables and all eight figures directly from the included measured summaries:

```bash
python scripts/write_revision_tex.py
python scripts/make_revision_figures.py
```

Figures are exported to `figure/revision/` as editable SVG, PDF and PNG. Their input hashes are recorded in `source_manifest.json`. Figures 1–6 concern the revised experiments; figures 7–8 are explicitly labeled diagnostics on the older split.

## Reproduce the paper experiments

The original corpus, its annotations and graph, large fitted caches, and the BGE checkpoint are not included. Dataset provenance and redistribution information remain incomplete in the author's source records. See [data/README.md](data/README.md) for exact schemas and the original input hashes. With the original five CSVs placed in `data/`, run:

```bash
python scripts/run_revision_experiments.py --seeds 17 29 43
```

The revised main experiment requires four CSVs: patents, document topics, the topic tree and graph edges. The fifth file, `retrieval_pairs.csv`, is needed for the older split diagnostics. `--prepare-only` prepares splits and pools without model fitting. A full run writes new files and replaces same-named summaries in `results/revision/`; use a fresh checkout to retain the bundled snapshot.

Reconstruct the historical model inputs before running the old-split appendix diagnostics:

```bash
python scripts/run_experiments.py
python scripts/revision_gate.py --run-old-split --maxiter 150
```

These are separate experiments. The old pipeline creates `results/fitted_models.pkl` and `results/pair_predictions.csv`, which the diagnostic entry point needs.
