# SpikeLoRA-TS R3: Reanalysis and Reproducibility Update

Prepared release tag: `v1.1-NCAA-R3`.

This update adds verified reanalysis of the retained corrected R2 records and original-dataset diagnostics. No forecasting model was retrained or re-evaluated, and no missing predictions were reconstructed. The original adapter and experimental programme are preserved.

## Added

- Original R3 reanalysis and condition-figure scripts.
- Seven byte-identical retained R2 numerical/configuration inputs, 17 approved derived CSV tables and the analysis-summary JSON.
- Dependency-aware aggregation and source-block sensitivity, contextual dense-reference gaps, adaptation calendar/distribution and optimization-exposure audits, joint accuracy/sparsity, horizon intervals and paired seed-level ramp summaries.
- Artifact availability, correction/scope notes, manuscript-output map, environment record, SHA-256 manifests and a no-training verification/packaging utility.

Fifteen derived CSVs are stored directly; the verifier materializes two approved tables locally using a lossless calendar-table archive and retained E2 metric records. Exact approved hashes are required. The curated package produced by `python reanalysis/verify_release.py --strict-bytes --build-package` contains all 17 plain CSVs.

## Corrected interpretation

The historical protocol remains `SpikeLoRA-TS-R2-corrected-v1`. R3 is not a renamed training protocol. Head-only similarity is descriptive, not equivalence. Hierarchical aggregation is not a fitted random-effects model. WP ramp grouping failed through non-finite-value handling; no corrected WP model-error strata are claimed. LIF-specific causal superiority and stage-specific encoding effects remain unestablished by the retained ablations.

## Preserved

The R2 tag/release, forecasting code, configuration, original numerical records and historical audit documents remain unchanged. Only the landing README/citation metadata are updated; separately identified R3 materials are added. Original data are not silently replaced.

## Availability and scope

The curated package excludes private correspondence, reviewer comments, author-only notes, third-party datasets and unavailable model/prediction binaries. Read `ARTIFACT_AVAILABILITY.md`. This public repository is not an anonymous review route. Software-release publication does not imply journal acceptance or independent verification of historical training.

The GitHub Release must be published separately; these notes alone do not create a tag or Release object.
