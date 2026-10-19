# R3 corrections, scope and preserved provenance

The original forecasting results and protocol are preserved. This document supersedes only the interpretations identified below in the historical R2 documentation.

## Statistical unit and strength of evidence

The original E2 numerical record contains 36 matched task/fraction/seed pairs and 12 seed-averaged conditions. These do not supply 36 or 12 independent data-source replications. Fractions reuse nested windows and base checkpoints, and SR/WS share source observations. R3 reports task/source-balanced log-ratio aggregation, whole-source sign flips and source omissions. This is not a fitted mixed-effects model. Sign-flip reference probabilities require independent, sign-symmetric source blocks; with only three sources they are exploratory, not confirmatory population evidence. Favourable-outcome frequency must not be equated with established error-magnitude superiority.

## Backbone and dense references

R3 verifies identical E1/E2 final test origins and reports contextual adapted-to-dense error ratios. Training histories differ. The E1 SNN is not the unadapted E2 backbone, and these ratios do not measure recovery from that checkpoint or establish dense-model parity.

## Adaptation fractions

The window-selection rule is unchanged. R3 reconstructs actual calendar coverage, descriptive target/covariate distances and optimization exposure from the original data and logs. Fraction changes jointly affect temporal composition and updates per epoch. These are observed differences, not a controlled demonstration that seasonal shift or optimization alone caused deterioration. No new seasonal-sampling experiment was conducted.

## LIF gating, placement and encoding

The E4 paired differences and seed-only t intervals describe the existing matched adapter comparison on a fixed test period. They do not isolate temporal LIF dynamics from a generic nonlinear sparsifier. No new static-gate control was trained. Head-only and two-layer means can be described as close; statistical equivalence or non-inferiority was not established. Continuous input uses a separately pretrained backbone, so the comparison concerns whole pipelines rather than an adaptation-only encoding effect. The continuous-LoRA factorial cell is not supplied. Input-event sparsity, gate sparsity and accuracy are reported separately; activity does not establish hardware energy savings.

## Conformal intervals and temporal uncertainty

Per-horizon coverage, width and recorded calibration quantiles are available from numerical logs. Their empirical trade-offs are task/regime dependent. Marginal horizon coverage is not simultaneous multi-horizon coverage, and chronological separation does not guarantee exchangeability. No complete calibration-residual distribution, rolling prediction diagnostic or observation-level interval score is reconstructed from aggregate records.

## Wind-power ramp diagnostic correction

The historical E1/E2 WP ramp rows contain non-finite thresholds and one all-origin group labelled `high`. The original diagnostic calculates `np.quantile` before removing non-finite ramp magnitudes. Reconstructing the recorded preprocessing identifies one valid forecast origin with an undefined raw current target (31 October 2018, 15:40). The inputs can remain valid after limited forward fill and all required future targets can be observed, while the raw current target saved for the ramp diagnostic is still missing. This contaminates both quantiles; comparisons with NaN fall through to the `high` label.

The R3 finite-target reconstruction yields 9,668 finite WP origins, thresholds approximately 222.37 and 799.08 kW, and group counts 3,223 / 3,222 / 3,223. These are dataset-only diagnostic values. **Corrected WP regime-specific model errors are not available from the retained summaries and are not reported.** The historical claim that coincident finite tertiles were established because of zero inflation is withdrawn.

Original SR/WS/EC regime memberships and errors are retained. Raw-target reconstruction can differ at numerical ties and must not be used to relabel archived model errors. The ramp measure is the maximum absolute change over the forecast horizons, not a signed ramp rate or a prospectively detected operating regime. Seed-level intervals are not temporal block-bootstrap intervals. The archived forecasting runner is kept byte-identical; this release does not silently patch and rerun its diagnostics.

## Reproducibility and AI assistance

The release verifies retained-record arithmetic and dataset diagnostics, not the historical execution of model training. Historical runtime paths are not distributed binary artifacts. R3 script/documentation preparation received ChatGPT assistance; computational checks are recorded explicitly and do not replace the authors' scientific responsibility. This repository release does not publish private reviewer correspondence or internal author-review notes.
