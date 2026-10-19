# Data preparation

The raw CSV files are not redistributed because they remain subject to their original licences. Place the four source files in one directory using the exact filenames listed below.

- `Palestine-Solar.csv`
- `Palestine-wind.csv`
- `Turky-Wind-power-Turbine.csv`
- `Moroco-power-consumption.csv`

The experiment runner validates explicit time, target, and feature schemas. It does not use automatic target guessing for reported experiments.

Expected targets:

| Task | Time column | Target | Resolution |
|---|---|---|---:|
| SR | `date` | `GHI` | 15 minutes |
| WS | `date` | `Wind Speed` | 15 minutes |
| WP | `date` | `LV ActivePower (kW)` | 10 minutes |
| EC | `Datetime` | `PowerConsumption_Zone1` | 10 minutes |

Use `data/R2_dataset_audit.csv` to verify the SHA-256 hashes, date ranges, feature order, physical horizons, and supervised-window counts. A hash mismatch indicates that the input file is not byte-identical to the data used for the reported experiments.
Dataset access and redistribution must follow the terms of the original repositories. The manuscript cites the original dataset records and describes the exact preprocessing protocol.
