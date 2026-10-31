# Public HUPD revision reproduction package

Article: Candidate Composition, Ancestor Scores, and Local Mixing in Public Patent Topic Ranking

Authors: Haijie Zhang and Weidong Liu, Inner Mongolia University.

Artifact purpose: reconstruct the public IPC-topic ranking comparisons, candidate diagnostics and figure/table estimates, while separating supplied-state score replay from fresh fitting. See the guide for tested hardware, dependencies, observed timings and interpretation limits.

This is a local revision package, not a hosted repository or DOI deposit.

Read `notes/hupd/reproduction_guide.md`. The paper and supplement are in `ACCESS_latex_template_20260513/`; the current reviewer response, completion matrix and upload instructions are in `notes/hupd/revision2/`. Earlier response and verification notes are retained as historical records, not current completion claims.

Create a Python 3.11 environment and install `requirements-hupd.txt`, then run `python scripts/reproduce_hupd.py`. With all 15 supplied scoring states, the default recovers public data, restores exact scoring representations, independently verifies scores, rebuilds summaries and compiles the paper without refitting archived models. Use `--plan-only` to inspect the selected actions. `--stage data|restore|fit|analyze|summaries|paper` selects a stage; `summaries` needs neither raw text nor representation caches. Explicit `--fresh-refit` preserves the full previous experiment under `output/hupd_refit_archives/` and fits all models anew; it uses substantial CPU time. A new project without prior states or outputs fits normally. Incomplete or mixed releases stop with a recovery message. The 45 loss-matched constant-intercept controls are post-outcome secondary analyses, their supplied results are verified on default replay and summarized from outputs alone, without changing frozen primary contrasts or main BestSimple selection.

Full raw HUPD texts and third-party encoder weights are recovered from pinned upstream URLs and hashes. Confidential data and prior private-data results are excluded. Large text-feature caches are rebuilt from public text; compact fitted membership/projection/graph states and the original public-input embeddings preserve exact score replay. Every restored array is fingerprint checked. Historical liblinear fits used an unspecified internal RNG. Supplied scoring states preserve exact replay, while future fresh fits now set random_state to the declared run seed and are separate measurements. On macOS, XGBoost may also require an OpenMP runtime such as libomp.

The scientific target is primary IPC topic-tree agreement on a sample selected by HUPD’s January 2016 filing_date field, whose legal filing dates are not uniformly verified, not independently judged legal/search relevance. Text component grouping is not true-family grouping. See the paper's Discussion for these and other scope limits. Read notes/hupd/revision2/reproduction_extensions.md for the expanded 24-search, 45-static-cell, 6-topology-fit, 9-gate-cell, 3-learned-score-mixing and 36-auxiliary-cell archive. Default supplied-output summaries never refit these experiments.
