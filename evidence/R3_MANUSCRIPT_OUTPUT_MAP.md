# R3 manuscript-to-output map

This map refers to the supplied R3 LaTeX source. Table/figure numbers may change if the manuscript is repaginated; the LaTeX labels below are the stable locators. It identifies numerical sources rather than redistributing the manuscript or reviewer correspondence. Except for Figure 5, computations are in `reanalysis/reanalyse_r3.py`.

| Manuscript element / LaTeX label | Numerical source under `reanalysis/results/` | Interpretation |
|---|---|---|
| Table 7, `tab:e2_stats`; Eq. 14, `eq:r3_hierarchy` | `R3_E2_paired_effects.csv`, `R3_E2_condition_effects.csv`, `R3_analysis_summary.json`, `R3_hierarchical_effects.csv`, `R3_source_block_sign_flips.csv` | Paired/condition arithmetic and dependency-preserving source sensitivity, not fitted random effects. |
| Figure 5, `fig:e2_relative` | `R3_E2_condition_effects.csv`; plotted by `make_condition_figure.py` | Three-seed method means. The generated filename remains `Fig4_E2_condition_effects_R3.pdf` for source compatibility. |
| Table 10, `tab:e4_r2`; Section 5.6 | `R3_E4_joint_accuracy_sparsity.csv`, `R3_E4_paired_seeds.csv`, `R3_analysis_summary.json` | Input/gate sparsity and existing matched ablation; t intervals describe seed variability. |
| Table 16, `tab:r3_hierarchy` | `R3_hierarchical_effects.csv`, `R3_source_block_sign_flips.csv` | Equal-task/equal-source estimates and source omissions. |
| Table 17, `tab:r3_dense` | `R3_dense_reference_context.csv` | Same-test-period dense references with different training histories; not E2 recovery. |
| Table 18, `tab:r3_calendar` | `R3_calendar_coverage.csv` | Actual origins, calendar spans, lookback/label extents and gaps to test. |
| Table 19, `tab:r3_distribution`; Section 5.2 | `R3_target_distribution_audit.csv`, `R3_covariate_distribution_audit.csv`, `R3_calendar_composition.csv` | Descriptive target/covariate/calendar composition, not causal inference. |
| Table 20, `tab:r3_updates` | `R3_optimization_exposure.csv` | Recorded epochs and derived minibatch-update counts. |
| Table 21, `tab:r3_e1_intervals` | `R3_E1_horizon_intervals.csv` | E1 horizon coverage/width/quantile summaries. |
| Table 22, `tab:r3_e2_intervals` | `R3_E2_horizon_intervals.csv` | E2 horizon coverage/width/quantile summaries. |
| Table 23, `tab:r3_ramp_groups`; Eq. 15, `eq:ramp` | `R3_archived_valid_ramp_groups.csv`, `R3_analysis_summary.json` | Archived SR/WS/EC groups; dataset-only WP finite-target correction. |
| Table 24, `tab:r3_ramp_uncertainty` | `R3_ramp_paired_seed_uncertainty.csv` | Paired seed differences on fixed retrospective regimes. |
| Table 25, `tab:r3_high_ramps` | `R3_ramp_paired_seed_uncertainty.csv` | High-change means across fractions/horizons; not independent horizon replications. |
| Section 5.4 regime interval discussion | `R3_ramp_interval_diagnostics.csv` | Retained regime-level empirical coverage and width. |

Original E1-E4 point forecasts, dense baselines, resources and cross-domain summaries remain in `results/` and the original R2 release. `reanalysis/inputs/` preserves exact copies of the seven records used by R3. Refer to `R3_INPUT_MANIFEST.json` for their R2 paths and hashes.
