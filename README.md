SpikeLoRA-SNN

This repository provides the official implementation and benchmark datasets for:

SpikeLoRA: Parameter-Efficient Adaptation for Spiking Time-Series Forecasting

SpikeLoRA introduces a low-rank adaptation framework tailored for Spiking Neural Networks (SNNs), enabling efficient fine-tuning under data-scarce and resource-constrained environments. The method integrates learnable activation sparsity via Leaky Integrate-and-Fire (LIF) neurons with low-rank parameter updates, allowing stable and energy-aware adaptation for multi-horizon forecasting tasks.

Included Datasets (Clean CSV Format)

The repository contains cleaned and standardized benchmark datasets used in the experiments:

Solar Radiation (SR) – Palestine Solar dataset

Wind Speed (WS) – Palestine Wind dataset

Wind Power (WP) – Turkey Wind Turbine dataset

Electricity Consumption (EC) – Morocco Power Consumption dataset

Each dataset:

Is provided in clean CSV format

Contains timestamped multivariate time-series data

Is preprocessed for reproducible forecasting experiments

 Research Scope

SpikeLoRA is evaluated on:

Multi-horizon forecasting (short and long horizon)

Few-shot in-domain adaptation

Cross-domain transfer scenarios

Comparison with Full Fine-Tuning (FullFT)

Comparison with ANN and Transformer baselines

The framework demonstrates that parameter-efficient adaptation can be effectively extended to spiking regression models, supporting deployment in renewable and smart energy systems.

 Key Contributions

Low-rank adaptation for spiking time-series models

Learnable activation sparsity through LIF gating

Data-efficient fine-tuning under domain shift

Public reproducible benchmark datasets

Energy-aware forecasting perspective

Also, the implementation of the proposed SpikeLoRA-TS framework, along with scripts required to reproduce the main experimental results

📎 Citation

If you use this repository or datasets, please cite:

@article{Ayasi2026SpikeLoRA,
  title={SpikeLoRA: Parameter-Efficient Spiking Neural Models for Multi-Horizon Forecasting in Renewable and Smart Energy Time Series},
  journal={Neural Computing and Applications},
  year={2026}
}
