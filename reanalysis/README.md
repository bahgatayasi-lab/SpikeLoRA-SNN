# R3 reanalysis of the retained SpikeLoRA-TS R2 numerical record

This package performs statistical arithmetic and original-dataset diagnostics only. It does not import a model framework, load trained weights, train a model, run forecasting inference, or generate substitute predictions.

## Reproduce

The `inputs/` directory contains the original per-run E1-E4 CSVs, the original E2 regime CSV, the dataset audit and the original configuration. Supply the four original dataset CSVs in a separate directory; the script verifies their byte-level SHA-256 hashes before analysis. The datasets are not redistributed in this package.

```sh
python reanalyse_r3.py --data-dir /path/to/original/datasets
python make_condition_figure.py
```

By default, derived outputs are written to `results/` and the plot is written one directory above this script as `Fig4_E2_condition_effects_R3.pdf`. Optional arguments `--input-dir` and `--out-dir` select alternative directories. Rerunning the script changes only its derived outputs, not original inputs. Use a fresh output directory to preserve a prior analysis snapshot.

Verified analysis environment: NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0; Matplotlib is used only to draw the condition-level figure. Record your own environment when reproducing. A second execution produced 17 byte-identical derived CSVs; the summary JSON additionally records input files and environment details.

## Interpretation boundaries

- Paired comparisons use task/fraction/seed matching. Conditions still share observations, nested windows and base checkpoints.
- Hierarchical effects average seeds/fractions within tasks and tasks within three source blocks; SR/WS are one related source. This is hierarchical aggregation, not a fitted random-effects model.
- The eight whole-source sign flips preserve all within-source groupings. Their p-value is an exploratory conditional reference requiring independent, sign-symmetric source blocks, not validated population inference. Equal-task and equal-source estimates and source omissions are reported.
- Paired t intervals describe seed variability on a fixed test period, require a distributional assumption that cannot be assessed with n=3, and are unadjusted for multiple comparisons. They are not temporal block-bootstrap intervals.
- E1 and E2 have verified identical final-test origins but different training histories. Dense-reference error ratios are contextual comparisons, not unadapted-E2 recovery estimates.
- Distribution summaries use one finite raw target per forecast origin. Wasserstein distances are descriptive marginal distances, not iid significance tests or causal explanations.
- Original SR/WS/EC ramp membership is preserved. Raw-value reconstruction near floating-point ties must not be used to relabel archived model errors.
- WP has one valid forecast origin with an undefined raw current target. Finite-target thresholds/counts can be reconstructed, but corrected model-error strata cannot be recovered from summaries; none are reported here.


The exact calculations, input hashes, numerical checks and environment are recorded in `results/R3_analysis_summary.json`. See `../evidence/R3_MANUSCRIPT_OUTPUT_MAP.md` for table and figure sources. The public release excludes the manuscript correspondence and internal author notes.

## Release verification

From the repository root run `python reanalysis/verify_release.py`. This checks the original-input copies, unchanged historical R2 files, distributed R3 file hashes and parsed scientific summary. `--build-package` additionally creates a curated ZIP and checksum sidecar in `dist/`; it does not train or evaluate a forecasting model. To check every distributed file byte-for-byte, including JSON serialization, use `--strict-bytes` before regenerating any outputs.

The locked dependency versions are in `../requirements_R3_analysis.txt`. `../ENVIRONMENT_R3_analysis.json` records the release-preparation check environment. The historical training environment remains in `../ENVIRONMENT_R2.txt`.
