## pool_diagnostics

Fixed candidate-list inventory, with each of 18,711 test queries counted once per profile. a, Exact random-order nDCG@10 expectations, with outer-fold points. b, Grade-3 candidate-count histogram. Zero-IDCG and all-grade-3 counts are reported in the accompanying main table. These list inventories have no fitting-run SD.

Source data: source_data/pool_diagnostics_folds.csv, source_data/pool_diagnostics_summary.csv, source_data/pool_diagnostics_grade3_histogram.csv.

## composition

Composition diagnostic on the same 1,500 feasible test queries (500 per outer fold), using the seed-101 models selected on stratified validation pools and a fixed-seed random query sample from each full feasibility intersection. a, Measured hierarchy and leaf blends share the identical lexical mixing alpha within each fold. b, Their paired difference with a 95% component-conditional bootstrap band; fitting uncertainty is excluded because only one complete fitting run is used. c, Analytical true-membership calibration, separate from measured model performance: the ancestor score equals the tree-defined target, while the leaf score distinguishes grade 3 and ties grades 0–2. The leaf reference averages uniformly random tie breaking; it is not the measured identifier-tie result. Remaining grade counts follow the stated deterministic allocation; at ten grade-3 candidates the leaf oracle reaches nDCG@10 one. The empirical reversal reflects estimated memberships and fixed lexical weights. The true-member oracle curve increases by resolving lower-grade ties as the top grade grows, which is a different quantity. Curve points reuse models and are not independent refits.

Source data: source_data/composition_measured.csv, source_data/composition_paired_effect.csv, source_data/composition_analytical_calibration.csv.

## composition_controls

Fixed-fit composition controls as differences from the ancestor blend, in percentage points. Every panel reuses the same random sample of 1,500 common-feasible queries and seed-101 stratified-validation-selected models. Points describe supplied grade compositions, not independent refits. They exclude fitting uncertainty. Random is a rate-matched Bernoulli neighbor mask, not random ranking order.

Source data: source_data/composition_controls_measured_differences.csv.

## information_effects

Paired explicit-score omitted minus informed differences on the three candidate pools. Diamonds show query-weighted means over five complete fitting runs; horizontal lines are 95% percentile intervals from 5,000 paired fitting-run/global-component bootstrap draws. Small points show the five whole-refit differences; they are not additional queries. Positive values favor the first named model. Uniform is the primary pool; retrieval and grade-stratified comparisons are secondary. All-query nDCG retains zero-IDCG queries at zero. Folds share a corpus and overlapping training sets; the intervals do not represent independent-corpus replication. No equivalence band or p-value star is shown.

Source data: source_data/information_effects_effects.csv, source_data/information_effects_refits.csv.

## graph_effects

Paired learned gate minus validation-selected simple operation differences on the three candidate pools. Diamonds show query-weighted means over five complete fitting runs; horizontal lines are 95% percentile intervals from 5,000 paired fitting-run/global-component bootstrap draws. Small points show the five whole-refit differences; they are not additional queries. Positive values favor the first named model. Uniform is the primary pool; retrieval and grade-stratified comparisons are secondary. All-query nDCG retains zero-IDCG queries at zero. Folds share a corpus and overlapping training sets; the intervals do not represent independent-corpus replication. No equivalence band or p-value star is shown.

Source data: source_data/graph_effects_effects.csv, source_data/graph_effects_refits.csv.

## selection_diagnostics

Validation selection and gate-retention diagnostics. a, Selected no-graph alpha across three folds × five fitting runs per pool. b, Selected rho values and their descriptive profile means. c, Expected retained permitted neighbor slots per training pair: small points are fold/run summaries and larger markers their unweighted descriptive mean. These fifteen choices are not independent datasets. Retention refers to training pair occurrences, not unique graph arcs or test-edge retention. Every graph method shares its fold/run/pool no-graph alpha. Constant probability is selected on a validation grid rather than fitted with the learned gate loss.

Source data: source_data/selection_diagnostics_selections.csv, source_data/selection_diagnostics_graph_off.csv, source_data/selection_diagnostics_support.csv.

## alpha_validation

Complete validation mixture curves. a–c, Lexical–ancestor alpha curves. d–f, Uniform/expected-loss/mean-loss/ListNet neighbor curves at the common selected alpha, expressed as nDCG@10 changes from rho zero in percentage points. Thin lines show each of the fifteen fold/run validation curves per pool; thick lines show their unweighted descriptive means. All eligible validation queries contribute within a curve. Validation data select the parameters; these curves are not test effects and their fifteen values are not independent-corpus replicates.

Source data: source_data/alpha_validation_validation_curves.csv, source_data/alpha_validation_rho_validation_curves.csv.

## grade_boundaries

All six unequal-grade boundaries. Points average five repeated refits and error bars show their descriptive SD, not standard errors. Only queries containing both grades contribute to each boundary. Ties receive half credit and the dotted line is 0.5. Categories are not connected by lines. These are secondary estimates on fixed supplied candidate lists.

Source data: source_data/grade_boundaries_all_methods.csv.

## learning_curves

Retuned pair-training coverage at nested budgets of 128, 256 and 512 queries (40 candidate pairs per training query). Representations, outer partitions and all validation/test candidate lists remain fixed to the seed-101 main run. Both information sets rerun their four-configuration grids at reduced budgets; the 512-query point reuses the identical main fit. Lines show test-query-weighted means over the three outer folds; small points are individual fold results, not independent fitting runs or a confidence interval. The dotted hierarchy reference uses the same seed-101 endpoint queries, independently checked against the saved main fit. This is a single-fitting-run coverage diagnostic and does not report five-refit SD.

Source data: source_data/learning_curves_folds.csv, source_data/learning_curves_weighted_means.csv, source_data/learning_curves_fixed_main_references.csv.

