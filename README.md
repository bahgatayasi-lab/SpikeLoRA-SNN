# SpikeLoRA-TS: Sparse Low-Rank Adaptation of Spiking Temporal Models

Reproducibility repository for **SpikeLoRA-TS: Sparse Low-Rank Adaptation of Spiking Temporal Models for Multi-Horizon Energy Forecasting**.

## Versioned evidence

- **Forecasting protocol:** `SpikeLoRA-TS-R2-corrected-v1`, unchanged.
- **Historical release:** [`v1.0-NCAA-R2`](https://github.com/bahgatayasi-lab/SpikeLoRA-SNN/releases/tag/v1.0-NCAA-R2), unchanged.
- **R3 update:** additional reanalysis of retained records and original-dataset diagnostics, prepared for the separate release `v1.1-NCAA-R3`. Check the Releases page to confirm publication; the presence of these files does not itself mean a GitHub Release has been published.

The original forecasting-specific LIF-gated adapter and E1-E4 experimental programme are retained. No forecasting model was retrained or re-evaluated for R3. 

## Verify the R3 records without training or original datasets

```sh
python reanalysis/verify_release.py --strict-bytes
```

This standard-library-only command checks the 17 approved derived CSV tables, scientific scripts, seven original input copies, and the preserved R2 records. Fifteen CSVs are stored directly. Two additional approved tables are materialized locally when absent: the calendar-composition CSV is decompressed losslessly from `reanalysis/retained_tables/`, and the E2 horizon table is calculated from the retained per-run metrics. Both must match the original approved CSV SHA-256 exactly. No model, network connection, credentials, or original datasets are needed for this verification. Existing different tables are not silently replaced.

To build a curated release ZIP and its checksum locally:

```sh
python reanalysis/verify_release.py --strict-bytes --build-package
```

The resulting ZIP includes all 17 plain CSVs. It excludes private manuscript correspondence, reviewer comments, internal author notes, third-party datasets, unavailable model binaries, and publication automation.

## Reproduce all R3 analyses from the original datasets, without training

Use Python 3.13 and the versions recorded in `requirements_R3_analysis.txt`. Obtain the four original datasets separately using `DATA_README.md`; exact dataset hashes are checked.

```sh
python -m pip install -r requirements_R3_analysis.txt
python reanalysis/reanalyse_r3.py --data-dir /path/to/original/datasets
python reanalysis/make_condition_figure.py
python reanalysis/verify_release.py
```

The unchanged analysis script performs statistical arithmetic and read-only data diagnostics. It does not import a model framework, load weights, train models, or generate replacement predictions. Its default output is `reanalysis/results/`; use `--out-dir /path/to/new/results` to preserve distributed outputs. The figure script reads the default output directory and creates manuscript Figure 5 under the historical filename `Fig4_E2_condition_effects_R3.pdf`.

Verification after regeneration allows only recorded-environment/key-order differences in the summary JSON; `--strict-bytes` requires exact JSON bytes too. The 17 CSVs reproduced byte-for-byte during revision preparation. This does not independently verify historical training runs.

## Evidence and interpretation

Original result counts are E1: 84, E2: 108, E3: 54, E4: 24. They are not counts of independent scientific replications. R3 emphasizes condition/task/source dependence, contextual dense-reference accuracy gaps, calendar and optimization exposure, joint encoding/gate sparsity, horizon-specific coverage and width, and paired seed-level ramp uncertainty.

Hierarchical aggregation is not a fitted mixed-effects model. Source-block sign-flip calculations are exploratory under stated assumptions. Head-only similarity is descriptive, not demonstrated equivalence. The ablations do not establish LIF-specific causal superiority or an encoding-only effect. Conformal coverage is not a deployment guarantee under drift.

The historical WP ramp diagnostic has a non-finite-value handling failure. Target-based finite thresholds/counts can be reconstructed; corrected WP model-error strata are not available and are not claimed.

See [R3 corrections and scope](evidence/R3_CORRECTIONS_AND_SCOPE.md), [artifact availability](ARTIFACT_AVAILABILITY.md), [manuscript-output mapping](evidence/R3_MANUSCRIPT_OUTPUT_MAP.md), and [analysis details](reanalysis/README.md). Historical R2 audit interpretations identified in the correction note are superseded; their original files are retained for provenance.

## Repository layout

`code/`, `configs/`, `results/`, and `ENVIRONMENT_R2.txt` preserve the original R2 forecasting materials. `reanalysis/` contains R3 analysis, retained inputs and derived records. The root manifests identify scientific file hashes, exact input-source copies, and unchanged R2 files. `ENVIRONMENT_R3_analysis.json` is separate from the historical training environment. `legacy/` is earlier development material, not corrected experimental evidence.

Historical `prediction_file` and `base_checkpoint` fields are runtime path references, not downloadable artifacts. Original prediction arrays, model checkpoints and complete calibration-residual arrays are not distributed. See `ARTIFACT_AVAILABILITY.md`.

## Original training commands are not the R3 reanalysis route

The archived runner and configuration remain unchanged. E1-E4 training commands retrain models; consult the [R2 README](https://github.com/bahgatayasi-lab/SpikeLoRA-SNN/blob/v1.0-NCAA-R2/README.md) for that historical workflow. Its CLI constructs a default configuration rather than loading the archived JSON: the recorded run enabled `save_predictions`, whereas the CLI default is false.


