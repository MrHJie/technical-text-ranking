# Technical Text Ranking

Experiments on public patent texts examining how **candidate composition, ancestor aggregation, and local neighbor mixing** affect topic-ranking results.

Companion paper: **Candidate Composition, Ancestor Scores, and Local Mixing in Public Patent Topic Ranking**.

Authors: Haijie Zhang and Weidong Liu, Inner Mongolia University.

## Overview

This project uses the public [HUPD](https://github.com/suzgunmirac/hupd) dataset. The analytical cohort contains 18,711 patent-text inputs grouped into 18,388 text components, covering 75 IPC subclasses, 24 classes, and 6 sections.

Shared IPC section, class, and subclass membership defines graded topic agreement. Each query is evaluated on a list of 40 candidates, with comparisons covering:

- **Candidate composition**: uniform sampling, retrieval-conditioned candidates, and grade-stratified sampling.
- **Hierarchical information**: leaf scores, ancestor aggregation, and feature controls for MLP and LambdaMART.
- **Local mixing**: uniform mixing, similarity weighting, constant retention probabilities, and learned gates.

The evaluation uses three component-grouped outer folds and five repeated refits. The main metric is nDCG@10, accompanied by ordering accuracy between grade levels. The target is topic agreement under a classification tree. These results do not directly measure examiner relevance, legal novelty, or whole-corpus retrieval effectiveness.

## Quick Start

Use Python 3.11 and run commands from the project root:

```bash
git clone https://github.com/MrHJie/technical-text-ranking.git
cd technical-text-ranking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-hupd.txt
```

Inspect the execution plan first:

```bash
python scripts/reproduce_hupd.py --plan-only
```

Common commands:

```bash
# Rebuild statistics, tables, and figures from saved experiment outputs
python scripts/reproduce_hupd.py --stage summaries

# Run the full pipeline, restoring saved scoring states or fitting from scratch
python scripts/reproduce_hupd.py

# Archive existing results, then fit a new experiment
python scripts/reproduce_hupd.py --fresh-refit
```

The `summaries` stage requires the complete matching saved outputs in `results/hupd/`. Source code alone is insufficient to rebuild the reported results. Full training requires substantial time and disk space. Compiling the paper requires LaTeX and `latexmk`. On macOS, XGBoost may also require `libomp`.

See the [reproduction guide](notes/hupd/reproduction_guide.md) for detailed steps, resource requirements, and saved-state recovery instructions.

## Data and Models

Patent texts come from HUPD. Text representations and scores use TF–IDF, BM25, and frozen PatentSBERTa. The public reproduction materials do not redistribute raw patent texts or third-party model weights. Download scripts record upstream versions and file checksums. Data and pretrained models remain subject to their respective upstream terms. See the [source audit](notes/hupd/revision2/source_audit.md) for provenance and licensing details.
