# SpikeLoRA-TS:  Reproducibility Repository

Protocol: `SpikeLoRA-TS-v1`

This repository provides the implementation, configurations, experimental
results, and reproducibility materials for SpikeLoRA-TS, a parameter-efficient
adaptation method for spiking temporal forecasting of renewable-energy and
smart-grid time series.
## Contents

- `code/spikelora_ts_r2_colab.py`: standalone experiment runner.
- `code/SpikeLoRA_TS_R2_Experiments_After_finish_execution.ipynb`: completed Colab notebook.
- `configs/R2_protocol_config.json`: exact model and training protocol.
- `data/R2_dataset_audit.csv`: explicit schemas, hashes, periods, frequencies, horizons, and split counts.
- `results/`: per-seed E1-E4 results and generated summaries.
- `evidence/`: claim audit and extended E2 statistical validation.
- `ENVIRONMENT_R2.txt`: verified runtime details and determinism caveat.
- `DATA_README.md`: data placement and validation instructions.

## Installation

In a fresh environment:

```bash
python -m pip install -r requirements_R2.txt
```

A CUDA-enabled PyTorch build appropriate to the local driver is required for GPU execution. The completed run used the PyTorch build supplied by Google Colab.

## Dataset audit and smoke test

```bash
python code/spikelora_ts_r2_colab.py audit \
  --data-dir /path/to/data \
  --output-dir /path/to/results

python code/spikelora_ts_r2_colab.py smoke \
  --data-dir /path/to/data \
  --output-dir /path/to/results
```

Confirm that the audit reports `PowerConsumption_Zone1` for EC before launching the full matrix.

## E1: in-domain forecasting baselines

```bash
python code/spikelora_ts_r2_colab.py e1 \
  --data-dir /path/to/data \
  --output-dir /path/to/results \
  --tasks SR,WS,WP,EC \
  --models mlp,lstm,gru,tcn,patchtst,itransformer,snn_tcn \
  --seeds 0,1,2
```

## E2: temporal few-shot adaptation

```bash
python code/spikelora_ts_r2_colab.py e2 \
  --data-dir /path/to/data \
  --output-dir /path/to/results \
  --tasks SR,WS,WP,EC \
  --fractions 0.1,0.2,0.5 \
  --seeds 0,1,2
```

## E4: representative mechanism/sensitivity ablation

```bash
python code/spikelora_ts_r2_colab.py e4 \
  --data-dir /path/to/data \
  --output-dir /path/to/results \
  --tasks SR \
  --fractions 0.1 \
  --seeds 0,1,2
```

## E3: pairwise transfer boundary tests

```bash
python code/spikelora_ts_r2_colab.py e3 \
  --data-dir /path/to/data \
  --output-dir /path/to/results \
  --source WS --target SR \
  --fractions 0.1,0.2,0.5 \
  --seeds 0,1,2

python code/spikelora_ts_r2_colab.py e3 \
  --data-dir /path/to/data \
  --output-dir /path/to/results \
  --source WS --target WP \
  --fractions 0.1,0.2,0.5 \
  --seeds 0,1,2
```

## Regenerate summaries

```bash
python code/spikelora_ts_r2_colab.py summaries \
  --output-dir /path/to/results
```

Completed experimental keys are appended to CSV files and skipped on rerun unless `--force` is supplied.

## Reproducible release

The results reported in the associated manuscript correspond to the archived
release listed on the GitHub Releases page. The `legacy/` directory contains
developmental scripts that are not required to reproduce the reported results.
Use the runner, configuration, and result files documented in this README.

## License

The software is distributed under the MIT License in `LICENSE`. Third-party datasets remain subject to their original licences and are not redistributed in the corrected release tree.
