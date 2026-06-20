# SpikeLoRA-SNN

Official implementation and benchmark datasets for:

**SpikeLoRA: Parameter-Efficient Spiking Neural Models for Multi-Horizon Forecasting in Renewable and Smart Energy Time Series**

SpikeLoRA-SNN provides a parameter-efficient adaptation framework for spiking time-series forecasting. The method combines a Spiking Neural Network Temporal Convolutional Network (SNN-TCN) backbone with Low-Rank Adaptation (LoRA) and a spike-gated SpikeLoRA adapter. Instead of updating the full forecasting model during adaptation, SpikeLoRA freezes the pretrained spiking backbone and trains only a small low-rank adaptation pathway, supporting stable and data-efficient fine-tuning under limited data and domain-shift conditions.

The repository is intended for reproducible experiments in renewable-energy and smart-grid forecasting, including in-domain benchmarking, few-shot adaptation, and cross-domain transfer.

## Repository contents

- `spikelora_ts_githup.py` — Colab-exported experimental source code for SpikeLoRA-SNN / SpikeLoRA-TS experiments.
- `spikelora_ts_Githup.ipynb` — optional Colab notebook version of the same experimental workflow, if included in the repository.
- `Palestine-Solar.csv` — cleaned Solar Radiation (SR) benchmark dataset.
- `Palestine-wind.csv` — cleaned Wind Speed (WS) benchmark dataset.
- `Turky-Wind-power-Turbine.csv` — cleaned Wind Power (WP) benchmark dataset.
- `Moroco-power-consumption.csv` — cleaned Electricity Consumption (EC) benchmark dataset.

The filename `spikelora_ts_githup.py` follows the current uploaded script name. For clarity, a future commit may rename it to `spikelora_ts_github.py` or `spikelora_ts_experimental_source_code.py`.

## What the code does

The script implements the main SpikeLoRA-SNN / SpikeLoRA-TS workflow:

1. loads renewable-energy and smart-energy time-series datasets;
2. detects or uses time and target columns;
3. builds chronological train/validation/test splits;
4. constructs sliding-window multi-horizon forecasting samples;
5. adds calendar covariates to encode periodic structure;
6. standardizes features using training-set statistics only;
7. trains dense neural baselines, including MLP, LSTM, GRU, TCN, and PatchTST-style Transformer models;
8. trains an SNN-TCN backbone with delta spike encoding and LIF neurons;
9. applies LoRA and SpikeLoRA adapters through a two-stage pretrain-then-adapt protocol;
10. evaluates RMSE, MAE, SMAPE, trainable-parameter ratio, adaptation time, and SpikeLoRA sparsity;
11. runs in-domain, few-shot, and cross-domain transfer experiments;
12. exports CSV result tables for reproducibility and later analysis.

## Datasets

The repository contains cleaned benchmark datasets used in the SpikeLoRA-SNN experiments.

| ID | Dataset | File | Typical target variable | Scope |
|---|---|---|---|---|
| SR | Solar Radiation | `Palestine-Solar.csv` | `GHI` | Solar irradiance and meteorological forecasting |
| WS | Wind Speed | `Palestine-wind.csv` | `Wind Speed` or similar wind-speed column | Meteorological wind forecasting |
| WP | Wind Power | `Turky-Wind-power-Turbine.csv` | `LV ActivePower (kW)` | Wind-turbine SCADA power forecasting |
| EC | Electricity Consumption | `Moroco-power-consumption.csv` | `PowerConsumption_Zone1` or similar load column | Smart-grid electricity-demand forecasting |

The public script includes automatic time-column and target-column detection. For fully reproducible experiments, check that the detected target column matches the intended target variable before running all experiments.

Each dataset:

- is provided in clean CSV format;
- contains timestamped multivariate time-series data;
- is processed chronologically to avoid future-data leakage;
- is standardized using training-set statistics only;
- is used for direct multi-horizon forecasting with horizons `[1, 2, 4, 8, 24]`.

## Research scope

SpikeLoRA-SNN is evaluated under three main experimental settings.

### E1: In-domain benchmarking

Models are trained and evaluated on each dataset independently using the same chronological data split. The comparison includes:

- MLP;
- LSTM;
- GRU;
- TCN;
- PatchTST-style Transformer;
- SNN-TCN;
- SNN-TCN + LoRA;
- SNN-TCN + SpikeLoRA.

### E2: Few-shot adaptation

A pretrained SNN-TCN backbone is adapted using only a fraction of the available training data. The script evaluates:

- Full fine-tuning (FullFT);
- LoRA adaptation;
- SpikeLoRA adaptation.

Few-shot fractions are `[0.1, 0.2, 0.5]`.

### E3: Cross-domain transfer

The script also evaluates leave-one-task-out and aligned cross-domain transfer settings. In these experiments, the model is pretrained on source energy tasks and adapted to a target task using an aligned union feature space. This setting is used to identify when parameter-efficient adaptation is sufficient and when severe representation shifts may still require fuller model adaptation.

## Method summary

SpikeLoRA-SNN uses an SNN-TCN backbone for multi-horizon forecasting. Continuous time-series inputs are converted into spike events using a delta encoder, where spikes are generated when temporal changes exceed a threshold. The encoded events are processed by LIF neurons and causal temporal convolutional blocks.

For parameter-efficient adaptation, LoRA introduces a low-rank update to a frozen linear layer. SpikeLoRA extends this idea by adding a LIF-gated low-rank activation pathway. This makes the adaptation pathway sparse and updates only a small number of parameters.

In the two-stage protocol:

1. **Stage A: Backbone pretraining** — train the SNN-TCN backbone using the available training data.
2. **Stage B: Adaptation** — freeze the backbone and train only the LoRA or SpikeLoRA adapter parameters. FullFT is included as a comparison mode and updates all parameters.

## Environment

The released script was exported from Google Colab and installs the main dependencies directly inside the notebook/script.

Core dependencies:

- Python
- PyTorch
- torchvision
- torchaudio
- SpikingJelly
- NumPy
- pandas
- scikit-learn
- tqdm
- matplotlib

The script automatically selects a CUDA device when available and otherwise falls back to CPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The experiments are software simulations of spiking-neural-network dynamics. No neuromorphic chip is required to reproduce the numerical results. Spike sparsity is reported as a hardware-agnostic proxy for event-driven efficiency, not as a direct measurement of chip-level energy consumption.

## Reproducibility settings

The main public script uses the following default settings across the main experiments.

| Setting | Value |
|---|---:|
| Random seed in public script | `0` by default; some E3 blocks include `[0, 1, 2]` |
| Train/validation/test split | `70% / 10% / 20%`, chronological |
| Lookback window | `96` time steps |
| Forecast horizons | `[1, 2, 4, 8, 24]` |
| Batch size | `128` |
| Optimizer | AdamW |
| Weight decay | `1e-4` |
| Full training / FullFT learning rate | `1e-3` |
| LoRA / SpikeLoRA learning rate | `5e-3` |
| LoRA rank | `8` |
| LoRA scaling alpha | `16` |
| SpikeLoRA LIF threshold | `0.05` in E2/E3; `0.1` in the E1 call in the current script |
| SNN simulation steps | `8` in E2/E3; `4` in the E1 call in the current script |
| E1 max epochs | `30` |
| E2 pretrain epochs | `30` |
| E2 adaptation epochs | `20` |
| E3 pretrain epochs | `30` in the full leave-one-task-out block |
| E3 adaptation epochs | `20` in the full leave-one-task-out block |
| Few-shot fractions | `[0.1, 0.2, 0.5]` |

For a camera-ready or archival release, it is recommended to pin exact package versions in a `requirements.txt` or `environment.yml` file.

## Quick start in Google Colab

1. Open `spikelora_ts_Githup.ipynb` or paste `spikelora_ts_githup.py` into a Colab notebook.
2. Upload the four dataset CSV files to the Colab working directory:
   - `Palestine-Solar.csv`
   - `Palestine-wind.csv`
   - `Turky-Wind-power-Turbine.csv`
   - `Moroco-power-consumption.csv`
3. Run the cells from top to bottom.
4. Check the printed dataset summary to confirm the detected time and target columns.
5. Download the generated CSV result files.

## Quick start locally

The current `.py` file was exported from Colab and contains notebook shell commands such as `!pip install`. For command-line execution, first remove those notebook-only lines or convert the notebook to a clean Python script.

Then install the dependencies and run:

```bash
pip install torch torchvision torchaudio
pip install spikingjelly numpy pandas scikit-learn tqdm matplotlib
python spikelora_ts_githup.py
```

A cleaner local setup can be created by adding a `requirements.txt` file and running:

```bash
pip install -r requirements.txt
python spikelora_ts_githup.py
```

## Expected output files

Depending on which experiment blocks are executed, the script can generate:

- `E1_results.csv` — in-domain benchmark results across datasets and models;
- `E2_fewshot_results.csv` — few-shot adaptation results for FullFT, LoRA, and SpikeLoRA;
- `E3_leave_one_task_out_aligned.csv` — cross-domain transfer results using aligned feature spaces.

Some summary tables are created as pandas DataFrames inside the script and printed or displayed in the notebook. They can be exported manually if needed.

## Relation to SpikeLoRA-X

SpikeLoRA-SNN is the original forecasting and adaptation repository. It provides the spiking forecasting backbone, the LoRA and SpikeLoRA adaptation mechanisms, and the cleaned benchmark datasets.

SpikeLoRA-X is not a replacement for this repository. SpikeLoRA-X keeps the SpikeLoRA forecasting backbone unchanged and adds a post-hoc Responsible-AI analysis layer for:

- horizon-wise attribution;
- attribution-fidelity testing;
- physical-plausibility checking;
- spike-activity and edge-readiness reporting.

SpikeLoRA-X repository:

https://github.com/bahgatayasi-lab/SpikeLoRA-X

## Citation

If you use this repository, please cite the SpikeLoRA-SNN / SpikeLoRA-TS paper and, when relevant, the SpikeLoRA-X CAEPIA/LNAI extension.

```bibtex
@article{ayasi2026spikelora_ts,
  title   = {SpikeLoRA: Parameter-Efficient Spiking Neural Models for Multi-Horizon Forecasting in Renewable and Smart Energy Time Series},
  author  = {Ayasi, Bahgat Waleed Deeb and coauthors},
  journal = {To be updated with final venue information},
  year    = {2026},
  note    = {Code and benchmark datasets: https://github.com/bahgatayasi-lab/SpikeLoRA-SNN}
}
```

For the explainability extension:

```bibtex
@inproceedings{ayasi2026spikelorax,
  title     = {SpikeLoRA-X: Explainable, Responsible, and Energy-Efficient Spiking Neural Networks for Multi-Horizon Renewable-Energy Forecasting at the Edge},
  author    = {Ayasi, Bahgat Waleed Deeb and coauthors},
  booktitle = {Proceedings of CAEPIA 2026},
  series    = {Lecture Notes in Artificial Intelligence},
  year      = {2026},
  note      = {Code: https://github.com/bahgatayasi-lab/SpikeLoRA-X}
}
```

Update the BibTeX entries with the final author list, venue, DOI, and page numbers once available.

## License

Add a license file before final public release. If no license is added, external users do not have clear permission to reuse, modify, or redistribute the code and datasets.

## Notes

- The cleaned datasets are provided for reproducibility. Dataset reuse should follow the licensing and access terms of the original data providers.
- The current public script is a Colab-style research script. For long-term reproducibility, consider splitting it into modules such as `data.py`, `models.py`, `train.py`, and `experiments.py`.
- The current script reports spike sparsity and trainable-parameter ratios as efficiency indicators. Direct energy measurements on neuromorphic hardware are outside the scope of the released software simulation.
