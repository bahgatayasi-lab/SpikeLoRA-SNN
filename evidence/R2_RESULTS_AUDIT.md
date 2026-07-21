# SpikeLoRA-TS R2 results audit

## Integrity and completeness

- Protocol: `SpikeLoRA-TS-R2-corrected-v1`, configuration hash `1c553ec25a0e`.
- E1: 84/84 expected runs (4 tasks x 7 models x 3 seeds); no duplicated experimental keys and no non-finite core metrics.
- E2: 108/108 expected runs (4 tasks x 3 fractions x 3 methods x 3 seeds); no non-finite core metrics.
- E3: 54/54 expected runs (2 transfer pairs x 3 fractions x 3 methods x 3 seeds).
- E4: 24/24 expected runs (8 variants x 3 seeds).
- Targets are explicit: SR=GHI, WS=Wind Speed, WP=LV ActivePower (kW), EC=PowerConsumption_Zone1.
- All runs use the same protocol identifier and configuration hash.

## Claims supported by E1

- PatchTST is best on EC and WP; TCN is best on SR; GRU is best on WS.
- iTransformer is second on EC and WP, but not best on SR or WS.
- SNN-TCN is the lowest-ranked accuracy baseline on all four tasks. The manuscript must not claim dense-model parity or state-of-the-art accuracy.
- The SNN-TCN has the smallest model among the seven E1 architectures (about 75k parameters) and about 79.2% encoder sparsity, but is slower on the Tesla T4 because eight dense simulation steps are executed. No GPU energy-saving claim is supported.

## Claims supported by E2

- SpikeLoRA-TS updates 1,112-1,160 parameters (1.46%-1.52%), versus approximately 75k for FullFT.
- Across all 36 matched SpikeLoRA-TS/LoRA task-fraction-seed pairs, SpikeLoRA-TS wins 25 (69.4%).
- Median relative RMSE improvement is 1.76% with a bootstrap 95% CI of [0.11%, 4.95%]. Mean relative improvement is 2.15%, with CI [-2.14%, 6.31%].
- At the run level, the exact two-sided sign test gives p=0.029, while magnitude-sensitive two-sided Wilcoxon tests do not reach 0.05 (p=0.074 for relative differences; p=0.181 for raw RMSE differences). After averaging the three seeds within each task-fraction cell, SpikeLoRA-TS wins 8/12 cells; the cell-level sign test is p=0.388 and the relative/raw Wilcoxon tests are p=0.077/p=0.733. The run-level sign result is therefore directional evidence, not a confirmatory independent-replication result.
- Benefits are concentrated in SR and WS. WP and EC are mixed. Therefore, the result supports a conditional regularization benefit, not universal superiority.
- Average gate sparsity is 60.7% (range 39.0%-79.8%). Mean encoder sparsity is 78.8%.
- On the Tesla T4, SpikeLoRA-TS is slower than LoRA (24.16 s versus 20.55 s average adaptation time) and has higher inference latency (0.165 versus 0.121 ms/sample). The practical measured advantage is parameter/optimizer-state storage, not dense-GPU speed.

## Claims supported by E3

- FullFT is best in four of six cross-domain mean conditions; LoRA and SpikeLoRA-TS each win one.
- WS-to-SR transfer consistently favors FullFT. WS-to-WP is mixed.
- Low-rank adaptation alone is not a reliable solution to strong cross-domain representation shift.

## Claims supported by E4

- Under matched rank-8 delta encoding, SpikeLoRA-TS improves RMSE by 3.43% over LoRA.
- Rank 16 gives a small further improvement (1.17% relative to rank 8) but almost doubles the trainable ratio to 2.99%.
- Head-only SpikeLoRA-TS is statistically similar in mean RMSE to the two-layer adapter while using 0.73% trainable parameters.
- Threshold 0.05 is best among 0.025, 0.05, and 0.10, but the three-seed intervals overlap.
- Continuous-input SpikeLoRA-TS is best (RMSE 61.91), about 10.0% below the default delta-input variant. Delta encoding therefore represents an accuracy-sparsity design choice rather than an accuracy-optimal choice.

## Uncertainty and robustness

- Nominal 95% split-conformal coverage is not uniformly achieved under temporal distribution shift. E1 run-level PICP ranges from 0.741 to 0.965; E2 ranges from 0.771 to 0.991.
- These intervals should be presented as empirical uncertainty diagnostics. Exchangeability is violated by chronological drift, so coverage cannot be described as guaranteed.
- Ramp-regime analysis is informative for SR, WS, and EC. WP has zero-inflated ramps that collapse the tertile thresholds and should not be used for three-regime conclusions.

## Required manuscript positioning

The defensible central claim is:

> SpikeLoRA-TS is a strict, sparse low-rank adaptation mechanism for an SNN-TCN that reduces trainable state to about 1.5% and provides a modest, task-dependent directional tendency relative to conventional LoRA in temporally shifted few-shot energy forecasting, while cross-domain transfer and dense-GPU efficiency remain limitations.
