# Technical Text Ranking

Code for the paper "Separating Topic Supervision, Hierarchical Scoring and Local Mixing in Technical Text Ranking".

Paper: TODO(arXiv/DOI link) · Contact: TODO(name, affiliation, email)

## What is in here

```
scripts/    entry points: demo, revision experiments, figures, tex writer
tests/      unit tests
data/       input CSVs (not in the repo, see Data availability)
results/    bundled measured summaries; new runs write here
figure/     figures; revision figures land in figure/revision/
```

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

All eight figures and the tables come from the summaries bundled in `results/revision/`, so this part needs no raw data:

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

## Appendix diagnostics (older split)

These need the historical model inputs, which are not shipped:

```bash
python scripts/run_experiments.py
python scripts/revision_gate.py --run-old-split --maxiter 150
```

The first command produces `results/fitted_models.pkl` and `results/pair_predictions.csv`; the second reads them.

## Data availability

The corpus, its annotations, the graph, the fitted caches and the BGE checkpoint are not in this repo. My records on where the data came from and under what terms it can be redistributed are incomplete, so I am not publishing it right now.

What that means if you want to check the work: the demo runs, and every table and figure can be rebuilt from the bundled summaries, but the models cannot be refitted from raw text. The summaries can only be checked against the hashes in `source_manifest.json`. If you need the raw inputs, write to me.

## Known issues

- Runtime and memory for a full run are not documented yet. TODO(实测: 轮次耗时、峰值内存)
- BGE encoding on CPU is slow; how slow depends on the batch size. TODO(实测数字)
- The fitted caches get large and are gitignored on purpose. TODO(体积)
- Raw-data steps have not been re-run end to end since the revision, so treat `run_experiments.py` as the least-tested path here.

## Citation

```bibtex
@inproceedings{TODO,
  title     = {Separating Topic Supervision, Hierarchical Scoring and Local Mixing in Technical Text Ranking},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

## License

No license file yet. TODO(代码与图表分别选什么许可，例如代码 MIT、图表 CC BY 4.0)

## Contact

TODO(name / affiliation / email)
