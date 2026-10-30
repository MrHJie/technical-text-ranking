# Technical Text Ranking

Source code and LaTeX sources for **Separating Topic Supervision, Hierarchical Scoring and Local Mixing in Technical Text Ranking**, by Haijie Zhang and Weidong Liu.

[中文说明](README_zh.md) · [Data format](data/README.md) · [Reproduction details](docs/REPRODUCIBILITY.md) · [GitHub upload guide](docs/GITHUB_UPLOAD_zh.md)

This repository contains the implementation used for the revised manuscript, its 26 compact result summaries, eight figures, and manuscript sources. The main experiment and gate implementation are preserved byte for byte from the recorded runs. The figure script has one portability change: its Matplotlib cache uses the system temporary directory.

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

These are separate experiments. The old pipeline creates `results/fitted_models.pkl` and `results/pair_predictions.csv`, which the diagnostic entry point needs. Full details and the optional BGE stage are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Protocol and saved results

The paper collapses 24,740 source records to 18,460 distinct effective inputs and forms 4,895 transitive family/input components. Three 70/15/15 component splits use seeds 17, 29 and 43. Each split has 1,200/250/350 training/validation/test queries, each with 20 lexical plus 20 random candidates outside the query component. Topic-derived grades are assigned after candidate selection.

The implementation includes training-only multiview character TF-IDF, BM25, component-grouped cross-fitting for topic predictions, matched flat and hierarchical controls, an MLP ensemble, LambdaMART, uniform neighbor mixing and three exact Bernoulli-mask gate objectives. See the manuscript and [reproduction details](docs/REPRODUCIBILITY.md) for equations and model selection.

| Method | Mean test nDCG@10 over three splits |
| --- | ---: |
| TF-IDF | 0.966813 |
| BM25 | 0.989873 |
| BGE-M3 zero-shot dense | 0.959042 |
| MLP ensemble | 0.994610 |
| LambdaMART | 0.990811 |
| Flat leaf | 0.998645 |
| Flat blend | 0.998639 |
| Flat pair ridge | 0.998571 |
| Primary hierarchy | 0.995377 |
| Hierarchy | 0.999122 |
| Uniform neighbors | 0.999140 |
| Gate expected / mean / ListNet, each | 0.999122 |

The three gates have the same per-query test nDCG@10 as the no-graph hierarchy on all three splits. These results do not establish a stable graph improvement. The near-ceiling scores concern topic-defined relevance in fixed candidate pools; they are not full-corpus retrieval or legal novelty measurements. Split standard deviations describe variation among alternative splits of one corpus.

## Manuscript sources

The current manuscript entry point is `ACCESS_latex_template_20260513/access.tex`. It reads the `revision_*.tex` sources; bibliography files are in the repository root. IEEE template resources retain their original notices and are separate from the research implementation.

With a TeX installation containing `latexmk` and BibTeX:

```bash
cd ACCESS_latex_template_20260513
LC_ALL=C LANG=C LC_CTYPE=C latexmk -pdf -interaction=nonstopmode -halt-on-error access.tex
```

The locale prefix avoids a Perl locale failure on some macOS installations. In Windows PowerShell, run the `latexmk` command without the three environment assignments.

## Repository layout

```text
scripts/                         Experiment, demo, figure and table entry points
tests/test_core.py               Mathematical and protocol checks
results/revision/                Compact recorded paper summaries
data/                            Schema and expected original input hashes
figure/revision/                 Eight editable figures and source manifest
ACCESS_latex_template_20260513/   Current manuscript and template resources
references*.bib                  Bibliography
docs/                            Reproduction and upload instructions
```

The CI workflow installs the core environment, runs the tests and synthetic demo, and rebuilds tables and figures. BGE inference and the original corpus experiments are separate from these checks. Repository authorship metadata is provided in `CITATION.cff`; no publication DOI or repository URL is invented.
