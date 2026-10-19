# Artifact availability and scope of reproducibility

The original forecasting experiments use protocol `SpikeLoRA-TS-R2-corrected-v1` and remain archived under `v1.0-NCAA-R2`. The R3 update adds reanalysis of retained experimental records; it does not introduce new forecasting-model training or inference.

## Available materials

- Original corrected R2 per-run numerical records, configurations, dataset audit and experiment runner.
- R3 analysis scripts, derived numerical outputs, checksums and a manuscript-output map.
- Dataset provenance and expected hashes. Third-party datasets are not redistributed in this release.

## Artifacts not distributed

The original prediction arrays, trained-model checkpoints and complete calibration-residual arrays are not included. Historical `prediction_file` and `base_checkpoint` paths retained in CSVs identify their original runtime locations; they are not downloadable files or evidence of their current availability.

The R3 scripts operate on retained numerical records and the original datasets. They cannot reconstruct individual forecasts or learned weights from aggregate errors. Analyses requiring unavailable artifacts, including corrected WP regime-specific model errors, temporal-block forecast uncertainty, complete calibration-versus-test residual distributions and true recovery from an unadapted E2 checkpoint, are not claimed to have been reproduced.

## Preservation and interpretation

The R2 tag, original numerical records and forecasting code are retained unchanged. R3 correction notes supersede specified historical interpretations, not the original experiment records. The WP ramp grouping issue is a non-finite-value diagnostic failure; the R3 target-based reconstruction does not constitute corrected model-error strata.

For the complete analytical scope, assumptions and reproduction instructions, see `reanalysis/README.md` and `evidence/R3_CORRECTIONS_AND_SCOPE.md`.
