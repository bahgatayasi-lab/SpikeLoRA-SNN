# R2 results manifest

| File | Type | Data rows / top-level keys | SHA-256 |
|---|---:|---:|---|
| `E1_R2_results.csv` | CSV | 84 | `bc4701c6f55f22ba76a4bd1498581a8b51b10973e5eadee140d28bf35f5c03da` |
| `E1_R2_summary_ci95.csv` | CSV | 28 | `3f310480d99e1fd3aa50eb5999c9d8be1008140b865e7ac3832fb30819faa80b` |
| `E1_R2_ramp_regime_analysis.csv` | CSV | 210 | `9b11fad473d1beb5c84c72b0d6b578032d0fb86d905471a72d90550e043ff10a` |
| `E1_R2_ramp_regime_summary_ci95.csv` | CSV | 70 | `891c41fc485e30282430d7c47a24355b3a7c25c61836b4ba3e2819561616f358` |
| `E1_R2_results_smoke.csv` | CSV | 6 | `99259e21a13a4d1f94f2c03db824a45a52121d5b4cf37b568e152b02c270226a` |
| `E2_R2_fewshot_results.csv` | CSV | 108 | `e0e985ce810cb91b9e03923934b746f45e8f034953fa055d95b2948662e33f9a` |
| `E2_R2_summary_ci95.csv` | CSV | 36 | `d5a9b52b82797727be0ce5815b581dd8ddeae68a0af7f6a1191e618d8f719a7c` |
| `E2_R2_SpikeLoRA_vs_LoRA_paired.csv` | CSV | 36 | `a0984903329e134cc3f68f07d4042b2fb0be4173202236e35323d528d393e467` |
| `E2_R2_SpikeLoRA_vs_LoRA_statistics.json` | JSON | 11 keys | `e93e8fa3142f71dc1b016f5e4f0f383bb087de2a8ef5cd5add9a5364ae3a1c7d` |
| `E2_R2_statistical_validation_extended.json` | JSON | 19 keys | `e090fc900fb3afd11829cad0aba1b24b79044e0277f091c1ea781de2e8ab872f` |
| `E2_R2_SpikeLoRA_vs_LoRA_condition_means.csv` | CSV | 12 | `fbbbccf01d53cee52215eda7c593605ee82ee811e376007b7a24e3622359a6f4` |
| `E2_R2_SpikeLoRA_vs_LoRA_condition_statistics.json` | JSON | 12 keys | `60cd9366d9d56b398ef78550272d9a7c678a3a702d32ba24ad810fd0fd81195c` |
| `E2_R2_ramp_regime_analysis.csv` | CSV | 270 | `b22a588d5dcd6a3c8dbc3873e06e573a3133eeed9818078683e6afe551c6f71c` |
| `E2_R2_ramp_regime_summary_ci95.csv` | CSV | 90 | `077b1af4353d9b41af4456cba6fb39caa3b7c6217525f9e282acd5465541c07f` |
| `E3_R2_pairwise_transfer_results.csv` | CSV | 54 | `08481b51320f1f3bc6b1e7f53c2799de68b299078023a29cf812d2b5157347de` |
| `E3_R2_pairwise_summary_ci95.csv` | CSV | 18 | `bfd44b1877bf2ff96d8f334de3de208cdb5fcbdc5aaedd2d3397ab21d8c39b1e` |
| `E4_R2_ablation_results.csv` | CSV | 24 | `ac37542b18d4b2d2e6c4d0bd7ac560e910feda23dd274623ae5348060a775beb` |
| `E4_R2_ablation_summary_ci95.csv` | CSV | 8 | `91feff5be0c1c6be00ab8fe6e279444647b5df5ba22116e28f40cbd09e3c7277` |
| `R2_dataset_audit.csv` | CSV | 4 | `0f65d88a04e4d894eb325f6c9cd18bae8bd6f840e05cbfd2065e58b31a09d4bb` |
| `R2_protocol_config.json` | JSON | 45 keys | `59d856726a9b6632be320d3501473fd92886f4b463cb90c5c813ccdd4e0e5ebb` |

Expected main-run completeness:

- E1: 84 rows (4 tasks x 7 models x 3 seeds).
- E2: 108 rows (4 tasks x 3 fractions x 3 methods x 3 seeds).
- E3: 54 rows (2 source-target pairs x 3 fractions x 3 methods x 3 seeds).
- E4: 24 rows (8 variants x 3 seeds).
