# -*- coding: utf-8 -*-
"""SpikeLoRA-TS R2 corrected and reproducible experiment runner.

This file is designed for Google Colab and local Python execution.  It replaces
notebook-global state with explicit dataset schemas, disjoint chronological
adaptation splits, strict adapter freezing, resumable CSV logging, faithful
channel-independent PatchTST and iTransformer baselines, per-horizon metrics,
resource measurements, and statistical summaries.

Protocol identifier: SpikeLoRA-TS-R2-corrected-v1 (2026-07-20)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Dataset

try:
    import psutil
except Exception:  # pragma: no cover - optional measurement only
    psutil = None


PROTOCOL_ID = "SpikeLoRA-TS-R2-corrected-v1"
PROTOCOL_DATE = "2026-07-20"

# Avoid severe CPU oversubscription in notebooks and small containers.  CUDA
# execution is unaffected; users may override this after importing the module.
if not torch.cuda.is_available():
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        torch.set_num_interop_threads(max(1, min(2, os.cpu_count() or 1)))
    except RuntimeError:
        pass


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    task: str
    filename: str
    time_col: str
    target_col: str
    feature_cols: Tuple[str, ...]
    frequency_minutes: int
    description: str


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "SR": DatasetSpec(
        task="SR",
        filename="Palestine-Solar.csv",
        time_col="date",
        target_col="GHI",
        feature_cols=(
            "Temperature", "Cloud Type", "Dew Point", "Relative Humidity",
            "Solar Zenith Angle", "Pressure", "Wind Direction", "Wind Speed", "GHI",
        ),
        frequency_minutes=15,
        description="Solar radiation / GHI forecasting (Palestine)",
    ),
    "WS": DatasetSpec(
        task="WS",
        filename="Palestine-wind.csv",
        time_col="date",
        target_col="Wind Speed",
        feature_cols=(
            "Temperature", "Cloud Type", "Dew Point", "Relative Humidity",
            "Pressure", "Wind Direction", "Wind Speed",
        ),
        frequency_minutes=15,
        description="Wind-speed forecasting (Palestine)",
    ),
    "WP": DatasetSpec(
        task="WP",
        filename="Turky-Wind-power-Turbine.csv",
        time_col="date",
        target_col="LV ActivePower (kW)",
        feature_cols=("Wind Speed (m/s)", "Wind Direction (°)", "LV ActivePower (kW)"),
        frequency_minutes=10,
        description="Wind-turbine active-power forecasting (Turkey)",
    ),
    "EC": DatasetSpec(
        task="EC",
        filename="Moroco-power-consumption.csv",
        time_col="Datetime",
        target_col="PowerConsumption_Zone1",
        feature_cols=("Temperature", "Humidity", "WindSpeed", "PowerConsumption_Zone1"),
        frequency_minutes=10,
        description="Electricity-consumption forecasting, Zone 1 (Morocco)",
    ),
}


@dataclass
class ExperimentConfig:
    protocol_id: str = PROTOCOL_ID
    lookback: int = 96
    horizons: Tuple[int, ...] = (1, 2, 4, 8, 24)
    calendar_features: Tuple[str, ...] = ("hour", "dayofweek", "month")
    forward_fill_limit: int = 2

    # E1: train / early-stopping validation / uncertainty calibration / test
    e1_split: Tuple[float, float, float, float] = (0.70, 0.05, 0.05, 0.20)

    # E2/E3: base train / base val / adaptation pool / adaptation val /
    # uncertainty calibration / test. These are disjoint chronological anchors.
    adaptation_split: Tuple[float, float, float, float, float, float] = (
        0.45, 0.05, 0.20, 0.05, 0.05, 0.20
    )
    fewshot_fractions: Tuple[float, ...] = (0.10, 0.20, 0.50)

    batch_size: int = 128
    num_workers: int = 0
    max_epochs_e1: int = 40
    max_epochs_base: int = 40
    max_epochs_adapt: int = 25
    patience: int = 7
    weight_decay: float = 1e-4
    lr_full: float = 1e-3
    lr_transformer: float = 5e-4
    lr_adapt_fullft: float = 1e-3
    lr_adapt_peft: float = 5e-3
    grad_clip: float = 1.0

    spike_steps: int = 8
    snn_channels: Tuple[int, ...] = (64, 64, 64)
    lif_threshold: float = 1.0
    lif_tau: float = 2.0
    delta_threshold: float = 0.05
    encoder_mode: str = "delta"  # delta or continuous

    lora_rank: int = 8
    lora_alpha: float = 16.0
    spikelora_threshold: float = 0.05
    adapter_targets: Tuple[str, ...] = ("in_proj", "head")

    deterministic: bool = True
    save_predictions: bool = False
    measure_latency_batches: int = 20

    # Dataset-scaled baseline sizes.
    mlp_hidden: int = 256
    rnn_hidden: int = 128
    rnn_layers: int = 2
    tcn_channels: int = 128
    tcn_levels: int = 3
    transformer_d_model: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 3
    transformer_ff: int = 256
    patch_len: int = 16
    patch_stride: int = 8
    dropout: float = 0.10

    def validate(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must contain positive integer lead times")
        if tuple(sorted(set(self.horizons))) != tuple(self.horizons):
            raise ValueError("horizons must be sorted and unique")
        if not math.isclose(sum(self.e1_split), 1.0, abs_tol=1e-8):
            raise ValueError("e1_split must sum to 1")
        if not math.isclose(sum(self.adaptation_split), 1.0, abs_tol=1e-8):
            raise ValueError("adaptation_split must sum to 1")
        if self.encoder_mode not in {"delta", "continuous"}:
            raise ValueError("encoder_mode must be 'delta' or 'continuous'")
        if any(t not in {"in_proj", "head"} for t in self.adapter_targets):
            raise ValueError("adapter_targets may contain only 'in_proj' and/or 'head'")


ALL_E1_MODELS: Tuple[str, ...] = (
    "mlp", "lstm", "gru", "tcn", "patchtst", "itransformer", "snn_tcn"
)
ADAPTATION_MODES: Tuple[str, ...] = ("fullft", "lora", "spikelora")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(cfg: ExperimentConfig) -> str:
    payload = stable_json(asdict(cfg)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------

@dataclass
class TaskSeries:
    task: str
    spec: DatasetSpec
    times: pd.DatetimeIndex
    feature_names: Tuple[str, ...]
    features: np.ndarray  # [N, F], short gaps may be forward-filled
    target: np.ndarray  # [N], observed target only; inserted rows remain NaN
    input_valid: np.ndarray  # [N], all input features available after limited ffill
    label_observed: np.ndarray  # [N], target was genuinely observed, not imputed
    target_feature_index: int
    audit: Dict[str, Any]


def _calendar_matrix(index: pd.DatetimeIndex, names: Sequence[str]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    cols: List[np.ndarray] = []
    out_names: List[str] = []
    for name in names:
        if name == "hour":
            cols.append(index.hour.to_numpy(dtype=np.float32) / 23.0)
            out_names.append("cal_hour")
        elif name == "dayofweek":
            cols.append(index.dayofweek.to_numpy(dtype=np.float32) / 6.0)
            out_names.append("cal_dayofweek")
        elif name == "month":
            cols.append(index.month.to_numpy(dtype=np.float32) / 12.0)
            out_names.append("cal_month")
        elif name == "dayofyear":
            cols.append(index.dayofyear.to_numpy(dtype=np.float32) / 366.0)
            out_names.append("cal_dayofyear")
        else:
            raise ValueError(f"Unsupported calendar feature: {name}")
    if not cols:
        return np.empty((len(index), 0), dtype=np.float32), tuple()
    return np.stack(cols, axis=1).astype(np.float32), tuple(out_names)


def load_task_series(
    task: str,
    data_dir: str | Path,
    cfg: ExperimentConfig,
    *,
    verbose: bool = True,
) -> TaskSeries:
    cfg.validate()
    if task not in DATASET_SPECS:
        raise KeyError(f"Unknown task {task!r}; valid tasks: {sorted(DATASET_SPECS)}")
    spec = DATASET_SPECS[task]
    path = Path(data_dir) / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Upload the four CSV files or set DATA_DIR to their directory."
        )

    raw = pd.read_csv(path)
    required = [spec.time_col, *spec.feature_cols]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{spec.filename} is missing required columns: {missing}")

    raw = raw[required].copy()
    raw[spec.time_col] = pd.to_datetime(raw[spec.time_col], errors="coerce", dayfirst=False)
    invalid_dates = int(raw[spec.time_col].isna().sum())
    if invalid_dates:
        raise ValueError(f"{spec.filename} contains {invalid_dates} invalid timestamps")
    if raw[spec.time_col].duplicated().any():
        duplicated = int(raw[spec.time_col].duplicated().sum())
        raise ValueError(f"{spec.filename} contains {duplicated} duplicate timestamps")

    for col in spec.feature_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values(spec.time_col).set_index(spec.time_col)
    original_index = pd.DatetimeIndex(raw.index)
    expected_index = pd.date_range(
        start=original_index.min(),
        end=original_index.max(),
        freq=f"{spec.frequency_minutes}min",
    )
    reindexed = raw.reindex(expected_index)

    # Preserve whether a target value was genuinely observed.  Short missing input
    # segments may be forward-filled, but imputed targets are never used as labels.
    label_observed = reindexed[spec.target_col].notna().to_numpy(dtype=bool)
    target_observed = reindexed[spec.target_col].to_numpy(dtype=np.float32)

    input_frame = reindexed[list(spec.feature_cols)].copy()
    inserted_rows = int(len(expected_index) - len(original_index))
    if cfg.forward_fill_limit > 0:
        input_frame = input_frame.ffill(limit=cfg.forward_fill_limit)
    input_valid = input_frame.notna().all(axis=1).to_numpy(dtype=bool)
    raw_features = input_frame.to_numpy(dtype=np.float32)

    calendar, calendar_names = _calendar_matrix(expected_index, cfg.calendar_features)
    features = np.concatenate([raw_features, calendar], axis=1).astype(np.float32)
    feature_names = tuple(spec.feature_cols) + calendar_names
    target_feature_index = feature_names.index(spec.target_col)

    observed_deltas = original_index.to_series().diff().dropna().dt.total_seconds().div(60.0)
    modal_interval = float(observed_deltas.mode().iloc[0]) if len(observed_deltas) else float("nan")
    remaining_invalid_input_rows = int((~input_valid).sum())
    observed_target_rows = int(label_observed.sum())

    audit = {
        "protocol_id": cfg.protocol_id,
        "task": task,
        "filename": spec.filename,
        "file_sha256": file_sha256(path),
        "description": spec.description,
        "time_col": spec.time_col,
        "target_col": spec.target_col,
        "feature_cols": stable_json(feature_names),
        "target_feature_index": target_feature_index,
        "original_rows": int(len(raw)),
        "regular_grid_rows": int(len(expected_index)),
        "inserted_missing_timestamps": inserted_rows,
        "forward_fill_limit_steps": int(cfg.forward_fill_limit),
        "remaining_invalid_input_rows": remaining_invalid_input_rows,
        "observed_target_rows": observed_target_rows,
        "start": str(expected_index.min()),
        "end": str(expected_index.max()),
        "configured_frequency_minutes": spec.frequency_minutes,
        "modal_observed_interval_minutes": modal_interval,
        "lookback_steps": cfg.lookback,
        "lookback_minutes": cfg.lookback * spec.frequency_minutes,
        "horizons_steps": stable_json(cfg.horizons),
        "horizons_minutes": stable_json([h * spec.frequency_minutes for h in cfg.horizons]),
    }

    if verbose:
        print(
            f"[{task}] target={spec.target_col!r}, rows={len(raw):,}, "
            f"regular_rows={len(expected_index):,}, inserted={inserted_rows:,}, "
            f"remaining_invalid_inputs={remaining_invalid_input_rows:,}, "
            f"frequency={spec.frequency_minutes} min"
        )

    return TaskSeries(
        task=task,
        spec=spec,
        times=expected_index,
        feature_names=feature_names,
        features=features,
        target=target_observed,
        input_valid=input_valid,
        label_observed=label_observed,
        target_feature_index=target_feature_index,
        audit=audit,
    )


def build_valid_anchors(series: TaskSeries, lookback: int, horizons: Sequence[int]) -> np.ndarray:
    """Return last-observed input indices t for windows X[t-L+1:t+1] -> y[t+h]."""
    max_h = int(max(horizons))
    n = len(series.times)
    if n <= lookback + max_h:
        raise ValueError(f"Task {series.task} is too short for lookback/horizons")

    anchors = np.arange(lookback - 1, n - max_h, dtype=np.int64)
    bad = (~series.input_valid).astype(np.int64)
    prefix = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(bad)])
    starts = anchors - lookback + 1
    input_bad_counts = prefix[anchors + 1] - prefix[starts]
    valid = input_bad_counts == 0
    for h in horizons:
        idx = anchors + int(h)
        valid &= series.label_observed[idx]
        valid &= np.isfinite(series.target[idx])
    return anchors[valid]


def split_anchors(
    anchors: np.ndarray,
    names: Sequence[str],
    ratios: Sequence[float],
    *,
    purge_steps: int = 0,
) -> Dict[str, np.ndarray]:
    if len(names) != len(ratios):
        raise ValueError("names and ratios must have equal length")
    if not math.isclose(float(sum(ratios)), 1.0, abs_tol=1e-8):
        raise ValueError("split ratios must sum to one")
    n = len(anchors)
    if n < len(names):
        raise ValueError("Not enough anchors for requested splits")
    boundaries = [0]
    cumulative = 0.0
    for ratio in ratios[:-1]:
        cumulative += float(ratio)
        boundaries.append(int(math.floor(n * cumulative)))
    boundaries.append(n)
    raw_parts = [anchors[boundaries[i]:boundaries[i + 1]] for i in range(len(names))]
    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        part = raw_parts[i]
        if purge_steps > 0 and i < len(names) - 1 and len(raw_parts[i + 1]):
            next_start = int(raw_parts[i + 1][0])
            # Keep an earlier forecast origin only when its farthest target lies
            # strictly before the next split's first forecast origin.
            part = part[part + int(purge_steps) < next_start]
        if len(part) == 0:
            raise ValueError(f"Split {name!r} is empty after purging")
        result[name] = part
    return result


@dataclass
class ScalingStats:
    x_mean: np.ndarray  # [F]
    x_std: np.ndarray  # [F]
    y_mean: np.ndarray  # [K]
    y_std: np.ndarray  # [K]

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
        }


def _rows_covered_by_windows(n_rows: int, anchors: np.ndarray, lookback: int) -> np.ndarray:
    diff = np.zeros(n_rows + 1, dtype=np.int64)
    starts = anchors - lookback + 1
    np.add.at(diff, starts, 1)
    np.add.at(diff, anchors + 1, -1)
    return np.cumsum(diff[:-1]) > 0


def fit_scaling_stats(
    series: TaskSeries,
    anchors: np.ndarray,
    lookback: int,
    horizons: Sequence[int],
) -> ScalingStats:
    row_mask = _rows_covered_by_windows(len(series.times), anchors, lookback)
    row_mask &= series.input_valid
    x = series.features[row_mask]
    if len(x) == 0:
        raise ValueError("No valid input rows available to fit scaling statistics")
    x_mean = np.nanmean(x, axis=0).astype(np.float32)
    x_std = np.nanstd(x, axis=0).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    y = np.stack([series.target[anchors + int(h)] for h in horizons], axis=1).astype(np.float32)
    y_mean = np.nanmean(y, axis=0).astype(np.float32)
    y_std = np.nanstd(y, axis=0).astype(np.float32)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
    return ScalingStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


class WindowDataset(Dataset):
    def __init__(
        self,
        series: TaskSeries,
        anchors: np.ndarray,
        lookback: int,
        horizons: Sequence[int],
        scaler: ScalingStats,
    ) -> None:
        self.series = series
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.lookback = int(lookback)
        self.horizons = tuple(int(h) for h in horizons)
        self.scaler = scaler

    def __len__(self) -> int:
        return int(len(self.anchors))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.anchors[idx])
        start = t - self.lookback + 1
        x = self.series.features[start:t + 1]
        x = ((x - self.scaler.x_mean) / self.scaler.x_std).astype(np.float32, copy=False)
        y_raw = np.asarray([self.series.target[t + h] for h in self.horizons], dtype=np.float32)
        y = ((y_raw - self.scaler.y_mean) / self.scaler.y_std).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


@dataclass
class PreparedData:
    series: TaskSeries
    splits: Dict[str, np.ndarray]
    scaler: ScalingStats

    def loader(
        self,
        split: str,
        cfg: ExperimentConfig,
        seed: int,
        *,
        shuffle: bool,
        anchors_override: Optional[np.ndarray] = None,
    ) -> DataLoader:
        anchors = self.splits[split] if anchors_override is None else anchors_override
        dataset = WindowDataset(self.series, anchors, cfg.lookback, cfg.horizons, self.scaler)
        return make_loader(dataset, cfg.batch_size, shuffle, seed, cfg.num_workers)


def prepare_e1_data(series: TaskSeries, cfg: ExperimentConfig) -> PreparedData:
    anchors = build_valid_anchors(series, cfg.lookback, cfg.horizons)
    splits = split_anchors(
        anchors, ("train", "val", "calib", "test"), cfg.e1_split,
        purge_steps=max(cfg.horizons)
    )
    scaler = fit_scaling_stats(series, splits["train"], cfg.lookback, cfg.horizons)
    return PreparedData(series=series, splits=splits, scaler=scaler)


def prepare_adaptation_data(series: TaskSeries, cfg: ExperimentConfig) -> PreparedData:
    anchors = build_valid_anchors(series, cfg.lookback, cfg.horizons)
    names = ("base_train", "base_val", "adapt_pool", "adapt_val", "calib", "test")
    splits = split_anchors(anchors, names, cfg.adaptation_split, purge_steps=max(cfg.horizons))
    scaler = fit_scaling_stats(series, splits["base_train"], cfg.lookback, cfg.horizons)
    return PreparedData(series=series, splits=splits, scaler=scaler)


def audit_all_datasets(
    data_dir: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    tasks: Sequence[str] = ("SR", "WS", "WP", "EC"),
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        series = load_task_series(task, data_dir, cfg, verbose=True)
        anchors = build_valid_anchors(series, cfg.lookback, cfg.horizons)
        e1 = split_anchors(
            anchors, ("train", "val", "calib", "test"), cfg.e1_split,
            purge_steps=max(cfg.horizons)
        )
        adapt = split_anchors(
            anchors,
            ("base_train", "base_val", "adapt_pool", "adapt_val", "calib", "test"),
            cfg.adaptation_split,
            purge_steps=max(cfg.horizons),
        )
        row = dict(series.audit)
        row.update({
            "valid_supervised_windows": int(len(anchors)),
            "e1_train_windows": int(len(e1["train"])),
            "e1_val_windows": int(len(e1["val"])),
            "e1_calibration_windows": int(len(e1["calib"])),
            "e1_test_windows": int(len(e1["test"])),
            "e2_base_train_windows": int(len(adapt["base_train"])),
            "e2_base_val_windows": int(len(adapt["base_val"])),
            "e2_adapt_pool_windows": int(len(adapt["adapt_pool"])),
            "e2_adapt_val_windows": int(len(adapt["adapt_val"])),
            "e2_calibration_windows": int(len(adapt["calib"])),
            "e2_test_windows": int(len(adapt["test"])),
        })
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "R2_dataset_audit.csv", index=False)
    with (output / "R2_protocol_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, sort_keys=True)
    return frame


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, horizons: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Metric shape mismatch: {y_true.shape} vs {y_pred.shape}")
    out: Dict[str, float] = {
        "test_rmse": rmse_np(y_true, y_pred),
        "test_mae": mae_np(y_true, y_pred),
        "test_smape": smape_np(y_true, y_pred),
    }
    for j, h in enumerate(horizons):
        out[f"rmse_h{h}"] = rmse_np(y_true[:, j], y_pred[:, j])
        out[f"mae_h{h}"] = mae_np(y_true[:, j], y_pred[:, j])
        out[f"smape_h{h}"] = smape_np(y_true[:, j], y_pred[:, j])
    return out


def conformal_interval_metrics(
    calibration_true: np.ndarray,
    calibration_pred: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    horizons: Sequence[int],
    *,
    alpha: float = 0.05,
    nonnegative: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Split-conformal absolute-residual intervals calibrated on a held-out block."""
    calibration_true = np.asarray(calibration_true, dtype=np.float64)
    calibration_pred = np.asarray(calibration_pred, dtype=np.float64)
    test_true = np.asarray(test_true, dtype=np.float64)
    test_pred = np.asarray(test_pred, dtype=np.float64)
    if calibration_true.shape != calibration_pred.shape:
        raise ValueError("Calibration truth/prediction shapes differ")
    if test_true.shape != test_pred.shape:
        raise ValueError("Test truth/prediction shapes differ")
    n_cal = calibration_true.shape[0]
    if n_cal < 2:
        raise ValueError("At least two calibration windows are required")
    residuals = np.abs(calibration_true - calibration_pred)
    # Finite-sample split-conformal quantile with the conservative 'higher' rule.
    quantile_level = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
    try:
        q = np.quantile(residuals, quantile_level, axis=0, method="higher")
    except TypeError:  # NumPy < 1.22
        q = np.quantile(residuals, quantile_level, axis=0, interpolation="higher")
    lower = test_pred - q.reshape(1, -1)
    upper = test_pred + q.reshape(1, -1)
    if nonnegative:
        lower = np.maximum(lower, 0.0)
    covered = (test_true >= lower) & (test_true <= upper)
    width = upper - lower
    out: Dict[str, float] = {
        "conformal_alpha": float(alpha),
        "conformal_calibration_n": int(n_cal),
        "picp95": float(np.mean(covered)),
        "mpiw95": float(np.mean(width)),
    }
    for j, h in enumerate(horizons):
        out[f"picp95_h{h}"] = float(np.mean(covered[:, j]))
        out[f"mpiw95_h{h}"] = float(np.mean(width[:, j]))
        out[f"conformal_q95_h{h}"] = float(q[j])
    return out, lower.astype(np.float32), upper.astype(np.float32), q.astype(np.float32)


# -----------------------------------------------------------------------------
# ANN / Transformer baselines
# -----------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    def __init__(self, lookback: int, n_features: int, n_horizons: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNNBaseline(nn.Module):
    def __init__(
        self,
        kind: str,
        n_features: int,
        n_horizons: int,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kind not in {"lstm", "gru"}:
            raise ValueError("kind must be lstm or gru")
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class Chomp1d(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.size == 0 else x[:, :, :-self.size].contiguous()


class CausalTemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.residual = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.out_act = nn.ReLU()
        for layer in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        z = self.drop2(self.act2(self.chomp2(self.conv2(z))))
        return self.out_act(z + self.residual(x))


class TCNBaseline(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        channels: int,
        levels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        c_in = n_features
        for level in range(levels):
            blocks.append(CausalTemporalBlock(c_in, channels, kernel=3, dilation=2**level, dropout=dropout))
            c_in = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        return self.head(z[:, :, -1])


class PatchTST(nn.Module):
    """Channel-independent PatchTST for target-variable forecasting.

    Each variate is patched and passed through a shared Transformer encoder.  The
    target variate's encoded patches are flattened into the multi-horizon head.
    This preserves the defining channel-independent and patch-token design.
    """

    def __init__(
        self,
        lookback: int,
        n_features: int,
        target_index: int,
        n_horizons: int,
        patch_len: int,
        stride: int,
        d_model: int,
        nhead: int,
        layers: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if lookback < patch_len:
            raise ValueError("lookback must be >= patch_len")
        self.lookback = lookback
        self.n_features = n_features
        self.target_index = target_index
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (lookback - patch_len) // stride
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reversible instance normalization, as commonly used with PatchTST.
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev

        # [B, L, C] -> [B, C, N, P]
        patches = normalized.transpose(1, 2).unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        b, c, n, p = patches.shape
        tokens = patches.contiguous().view(b * c, n, p)
        z = self.patch_proj(tokens) + self.position[:, :n, :]
        z = self.norm(self.encoder(z))
        z = z.view(b, c, n, -1)
        target_repr = z[:, self.target_index, :, :].reshape(b, -1)
        output = self.head(target_repr)
        target_mean = means[:, 0, self.target_index].unsqueeze(-1)
        target_std = stdev[:, 0, self.target_index].unsqueeze(-1)
        return output * target_std + target_mean


class ITransformer(nn.Module):
    """Core iTransformer: variates are tokens and attention operates across them."""

    def __init__(
        self,
        lookback: int,
        target_index: int,
        n_horizons: int,
        d_model: int,
        nhead: int,
        layers: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.target_index = target_index
        self.token_projection = nn.Linear(lookback, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The official iTransformer formulation normalizes each variate over the
        # lookback window, applies attention over variate tokens, then reverses
        # the normalization for the predicted target variate.
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev
        tokens = normalized.transpose(1, 2)  # [B, C, L]
        z = self.token_projection(tokens)
        z = self.norm(self.encoder(z))
        all_outputs = self.projector(z)  # [B, C, K]
        target_output = all_outputs[:, self.target_index, :]
        target_mean = means[:, 0, self.target_index].unsqueeze(-1)
        target_std = stdev[:, 0, self.target_index].unsqueeze(-1)
        return target_output * target_std + target_mean


# -----------------------------------------------------------------------------
# Explicit LIF and SpikeLoRA implementation (no hidden state between sequences)
# -----------------------------------------------------------------------------

class _SigmoidSurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sig = torch.sigmoid(alpha * x)
        grad = alpha * sig * (1.0 - sig)
        return grad_output * grad, None


def surrogate_spike(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    return _SigmoidSurrogateSpike.apply(x, float(alpha))


class LIFNode(nn.Module):
    """LIF node matching the common tau=2, decay-input, hard-reset formulation."""

    def __init__(
        self,
        threshold: float = 1.0,
        tau: float = 2.0,
        v_reset: float = 0.0,
        surrogate_alpha: float = 4.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("LIF tau must be > 1")
        self.threshold = float(threshold)
        self.tau = float(tau)
        self.v_reset = float(v_reset)
        self.surrogate_alpha = float(surrogate_alpha)
        self.detach_reset = bool(detach_reset)
        self.v: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.full_like(x, self.v_reset)
        h = self.v + (x - (self.v - self.v_reset)) / self.tau
        spike = surrogate_spike(h - self.threshold, self.surrogate_alpha)
        reset_spike = spike.detach() if self.detach_reset else spike
        self.v = h * (1.0 - reset_spike) + self.v_reset * reset_spike
        return spike


def reset_spiking_state(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, LIFNode):
            submodule.reset_state()


class ActivityTracker:
    def __init__(self) -> None:
        self.zeros = 0
        self.total = 0

    def reset(self) -> None:
        self.zeros = 0
        self.total = 0

    def update(self, tensor: torch.Tensor) -> None:
        with torch.no_grad():
            self.zeros += int((tensor == 0).sum().item())
            self.total += int(tensor.numel())

    @property
    def sparsity_pct(self) -> float:
        return float(100.0 * self.zeros / self.total) if self.total else float("nan")


class SpikeEncoder(nn.Module):
    def __init__(self, mode: str = "delta", threshold: float = 0.05):
        super().__init__()
        if mode not in {"delta", "continuous"}:
            raise ValueError("SpikeEncoder mode must be delta or continuous")
        self.mode = mode
        self.threshold = float(threshold)
        self.activity = ActivityTracker()

    def reset_activity(self) -> None:
        self.activity.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "continuous":
            encoded = x
        else:
            dx = x[:, 1:, :] - x[:, :-1, :]
            events = (dx.abs() > self.threshold).to(dtype=x.dtype)
            encoded = torch.cat([torch.zeros_like(x[:, :1, :]), events], dim=1)
        self.activity.update(encoded)
        return encoded


class SpikingTCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        channels: Sequence[int] = (64, 64, 64),
        simulation_steps: int = 8,
        lif_threshold: float = 1.0,
        lif_tau: float = 2.0,
        encoder_mode: str = "delta",
        delta_threshold: float = 0.05,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.simulation_steps = int(simulation_steps)
        self.encoder = SpikeEncoder(mode=encoder_mode, threshold=delta_threshold)
        self.in_proj = nn.Linear(n_features, int(channels[0]))
        self.lif = LIFNode(threshold=lif_threshold, tau=lif_tau, detach_reset=True)
        blocks: List[nn.Module] = []
        c_in = int(channels[0])
        for i, c_out in enumerate(channels):
            blocks.append(CausalTemporalBlock(c_in, int(c_out), kernel=3, dilation=2**i, dropout=dropout))
            c_in = int(c_out)
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(c_in, n_horizons)

    def reset_activity_stats(self) -> None:
        self.encoder.reset_activity()
        for module in self.modules():
            if isinstance(module, SpikeLoRALinear):
                module.reset_activity()

    def activity_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {"encoder_sparsity_pct": self.encoder.activity.sparsity_pct}
        adapters = [m for m in self.modules() if isinstance(m, SpikeLoRALinear)]
        if adapters:
            zeros = sum(m.activity.zeros for m in adapters)
            total = sum(m.activity.total for m in adapters)
            stats["spikelora_sparsity_pct"] = float(100.0 * zeros / total) if total else float("nan")
            for i, adapter in enumerate(adapters):
                stats[f"spikelora_adapter_{i}_sparsity_pct"] = adapter.activity.sparsity_pct
        else:
            stats["spikelora_sparsity_pct"] = float("nan")
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reset_spiking_state(self)
        encoded = self.encoder(x)
        prediction: Optional[torch.Tensor] = None
        for _ in range(self.simulation_steps):
            projected = self.in_proj(encoded)
            spikes = self.lif(projected)
            hidden = self.tcn(spikes.transpose(1, 2))[:, :, -1]
            step_prediction = self.head(hidden)
            prediction = step_prediction if prediction is None else prediction + step_prediction
        assert prediction is not None
        return prediction / float(self.simulation_steps)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank = self.dropout(x) @ self.A.t()
        return self.base(x) + self.scale * (low_rank @ self.B.t())


class SpikeLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        gate_threshold: float,
        lif_tau: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("SpikeLoRA rank must be >= 1")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        for p in self.base.parameters():
            p.requires_grad = False
        self.gate = LIFNode(threshold=gate_threshold, tau=lif_tau, detach_reset=True)
        self.activity = ActivityTracker()

    def reset_activity(self) -> None:
        self.activity.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank = self.dropout(x) @ self.A.t()
        gate = self.gate(low_rank)
        self.activity.update(gate)
        sparse_low_rank = gate * low_rank
        return self.base(x) + self.scale * (sparse_low_rank @ self.B.t())


def freeze_all(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class SpikingTCNAdapters(nn.Module):
    """Strict PEFT wrapper: only A/B matrices are trainable."""

    def __init__(
        self,
        base_model: SpikingTCN,
        mode: str,
        rank: int,
        alpha: float,
        gate_threshold: float,
        lif_tau: float,
        targets: Sequence[str],
    ) -> None:
        super().__init__()
        if mode not in {"lora", "spikelora"}:
            raise ValueError("mode must be lora or spikelora")
        self.base = base_model
        self.mode = mode
        self.targets = tuple(targets)
        freeze_all(self.base)

        for name in self.targets:
            layer = getattr(self.base, name)
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"Adapter target {name!r} is not nn.Linear")
            if mode == "lora":
                wrapped: nn.Module = LoRALinear(layer, rank=rank, alpha=alpha)
            else:
                wrapped = SpikeLoRALinear(
                    layer,
                    rank=rank,
                    alpha=alpha,
                    gate_threshold=gate_threshold,
                    lif_tau=lif_tau,
                )
            setattr(self.base, name, wrapped)

        trainable_names = [name for name, p in self.named_parameters() if p.requires_grad]
        illegal = [name for name in trainable_names if not (name.endswith(".A") or name.endswith(".B"))]
        if illegal:
            raise RuntimeError(f"Strict PEFT violation; non-adapter parameters are trainable: {illegal}")
        if not trainable_names:
            raise RuntimeError("No adapter parameters were made trainable")

    def reset_activity_stats(self) -> None:
        self.base.reset_activity_stats()

    def activity_stats(self) -> Dict[str, float]:
        return self.base.activity_stats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


# -----------------------------------------------------------------------------
# Model building, training, resource measurement
# -----------------------------------------------------------------------------

def count_total_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def serialized_model_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return float(buffer.tell() / (1024.0**2))


def trainable_parameter_size_mb(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return float(total_bytes / (1024.0**2))


def build_model(
    model_name: str,
    n_features: int,
    target_index: int,
    cfg: ExperimentConfig,
) -> nn.Module:
    k = len(cfg.horizons)
    if model_name == "mlp":
        return MLPBaseline(cfg.lookback, n_features, k, cfg.mlp_hidden, cfg.dropout)
    if model_name == "lstm":
        return RNNBaseline("lstm", n_features, k, cfg.rnn_hidden, cfg.rnn_layers, cfg.dropout)
    if model_name == "gru":
        return RNNBaseline("gru", n_features, k, cfg.rnn_hidden, cfg.rnn_layers, cfg.dropout)
    if model_name == "tcn":
        return TCNBaseline(n_features, k, cfg.tcn_channels, cfg.tcn_levels, cfg.dropout)
    if model_name == "patchtst":
        return PatchTST(
            cfg.lookback,
            n_features,
            target_index,
            k,
            cfg.patch_len,
            cfg.patch_stride,
            cfg.transformer_d_model,
            cfg.transformer_heads,
            cfg.transformer_layers,
            cfg.transformer_ff,
            cfg.dropout,
        )
    if model_name == "itransformer":
        return ITransformer(
            cfg.lookback,
            target_index,
            k,
            cfg.transformer_d_model,
            cfg.transformer_heads,
            cfg.transformer_layers,
            cfg.transformer_ff,
            cfg.dropout,
        )
    if model_name == "snn_tcn":
        return SpikingTCN(
            n_features,
            k,
            channels=cfg.snn_channels,
            simulation_steps=cfg.spike_steps,
            lif_threshold=cfg.lif_threshold,
            lif_tau=cfg.lif_tau,
            encoder_mode=cfg.encoder_mode,
            delta_threshold=cfg.delta_threshold,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model {model_name!r}")


def reset_activity_stats(model: nn.Module) -> None:
    if hasattr(model, "reset_activity_stats"):
        model.reset_activity_stats()  # type: ignore[attr-defined]


def get_activity_stats(model: nn.Module) -> Dict[str, float]:
    if hasattr(model, "activity_stats"):
        return dict(model.activity_stats())  # type: ignore[attr-defined]
    return {"encoder_sparsity_pct": float("nan"), "spikelora_sparsity_pct": float("nan")}


def _batch_to_device(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    max_batches: Optional[int] = None,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_count += int(x.size(0))
    if total_count == 0:
        raise RuntimeError("Training loader produced zero samples")
    return total_loss / total_count


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = _batch_to_device(batch, device)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        total_loss += float(loss.item()) * x.size(0)
        total_count += int(x.size(0))
    if total_count == 0:
        raise RuntimeError("Validation loader produced zero samples")
    return total_loss / total_count


@dataclass
class FitStats:
    best_epoch: int
    best_val_loss: float
    epochs_ran: int
    train_time_sec: float
    peak_gpu_memory_mb: float
    peak_process_rss_mb: float
    total_params: int
    trainable_params: int
    total_model_size_mb: float
    trainable_parameter_size_mb: float


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    *,
    learning_rate: float,
    max_epochs: int,
    patience: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
) -> Tuple[nn.Module, FitStats]:
    device = get_device()
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, (patience or cfg.patience) // 2), min_lr=1e-6
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process(os.getpid()) if psutil is not None else None
    peak_rss = process.memory_info().rss if process is not None else 0

    best_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    actual_patience = cfg.patience if patience is None else patience
    start = time.perf_counter()
    epochs_ran = 0

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        train_one_epoch(
            model, train_loader, optimizer, device, cfg.grad_clip, max_batches=max_train_batches
        )
        val = validation_loss(model, val_loader, device, max_batches=max_val_batches)
        scheduler.step(val)
        if process is not None:
            peak_rss = max(peak_rss, process.memory_info().rss)

        if val < best_loss - 1e-10:
            best_loss = val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= actual_patience:
            break

    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    peak_gpu = (
        float(torch.cuda.max_memory_allocated() / (1024.0**2)) if torch.cuda.is_available() else 0.0
    )
    peak_rss_mb = float(peak_rss / (1024.0**2)) if peak_rss else float("nan")
    stats = FitStats(
        best_epoch=best_epoch,
        best_val_loss=float(best_loss),
        epochs_ran=epochs_ran,
        train_time_sec=float(elapsed),
        peak_gpu_memory_mb=peak_gpu,
        peak_process_rss_mb=peak_rss_mb,
        total_params=count_total_params(model),
        trainable_params=count_trainable_params(model),
        total_model_size_mb=serialized_model_size_mb(model),
        trainable_parameter_size_mb=trainable_parameter_size_mb(model),
    )
    return model, stats


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    scaler: ScalingStats,
    horizons: Sequence[int],
    *,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Dict[str, float]]:
    device = get_device()
    model.to(device)
    model.eval()
    reset_activity_stats(model)
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = _batch_to_device(batch, device)
        pred = model(x)
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    if not preds:
        raise RuntimeError("Evaluation loader produced zero samples")
    pred_scaled = np.concatenate(preds, axis=0)
    true_scaled = np.concatenate(trues, axis=0)
    pred_raw = pred_scaled * scaler.y_std + scaler.y_mean
    true_raw = true_scaled * scaler.y_std + scaler.y_mean
    metrics = metric_dict(true_raw, pred_raw, horizons)
    activity = get_activity_stats(model)
    return metrics, true_raw, pred_raw, activity


@torch.no_grad()
def measure_inference_latency_ms(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int,
) -> Tuple[float, float]:
    device = get_device()
    model.to(device)
    model.eval()
    batches: List[torch.Tensor] = []
    for i, (x, _) in enumerate(loader):
        batches.append(x.to(device, non_blocking=True))
        if i + 1 >= max_batches:
            break
    if not batches:
        return float("nan"), float("nan")

    # Warm-up.
    for x in batches[: min(3, len(batches))]:
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = 0.0
    samples = 0
    for x in batches:
        start = time.perf_counter()
        _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        samples += int(x.size(0))
    per_sample_ms = 1000.0 * elapsed / max(1, samples)
    per_batch_ms = 1000.0 * elapsed / max(1, len(batches))
    return float(per_sample_ms), float(per_batch_ms)


def fit_stats_to_row(stats: FitStats) -> Dict[str, Any]:
    row = asdict(stats)
    row["trainable_ratio"] = stats.trainable_params / max(1, stats.total_params)
    return row


# -----------------------------------------------------------------------------
# Resumable result logging
# -----------------------------------------------------------------------------

def _normalise_key_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def result_key(row: Mapping[str, Any], key_cols: Sequence[str]) -> Tuple[str, ...]:
    return tuple(_normalise_key_value(row.get(col, "")) for col in key_cols)


def load_existing_keys(path: Path, key_cols: Sequence[str]) -> set[Tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    return {result_key(rec, key_cols) for rec in frame.to_dict(orient="records")}


def append_result(path: Path, row: Mapping[str, Any], key_cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([dict(row)])
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
        combined["__key"] = combined.apply(
            lambda r: "||".join(_normalise_key_value(r.get(c, "")) for c in key_cols), axis=1
        )
        combined = combined.drop_duplicates("__key", keep="last").drop(columns="__key")
    else:
        combined = new
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(path)


def base_result_metadata(
    task: str,
    seed: int,
    cfg: ExperimentConfig,
    prepared: PreparedData,
) -> Dict[str, Any]:
    spec = prepared.series.spec
    return {
        "protocol_id": cfg.protocol_id,
        "protocol_date": PROTOCOL_DATE,
        "config_hash": config_fingerprint(cfg),
        "task": task,
        "seed": int(seed),
        "target_col": spec.target_col,
        "time_col": spec.time_col,
        "frequency_minutes": spec.frequency_minutes,
        "lookback_steps": cfg.lookback,
        "lookback_minutes": cfg.lookback * spec.frequency_minutes,
        "horizons_steps": stable_json(cfg.horizons),
        "horizons_minutes": stable_json([h * spec.frequency_minutes for h in cfg.horizons]),
        "n_features": len(prepared.series.feature_names),
        "feature_names": stable_json(prepared.series.feature_names),
        "target_feature_index": prepared.series.target_feature_index,
        "data_file_sha256": prepared.series.audit["file_sha256"],
        "device": str(get_device()),
        "torch_version": torch.__version__,
    }


def save_predictions(
    output_dir: Path,
    experiment: str,
    identity: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Sequence[int],
    *,
    lower95: Optional[np.ndarray] = None,
    upper95: Optional[np.ndarray] = None,
    anchors: Optional[np.ndarray] = None,
    series: Optional[TaskSeries] = None,
) -> str:
    pred_dir = output_dir / "predictions" / experiment
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{identity}.npz"
    payload: Dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "horizons": np.asarray(horizons, dtype=np.int64),
    }
    if lower95 is not None and upper95 is not None:
        payload["lower95"] = lower95
        payload["upper95"] = upper95
    if anchors is not None and series is not None:
        anchors = np.asarray(anchors, dtype=np.int64)
        if len(anchors) < len(y_true):
            raise ValueError("Prediction count exceeds available anchor count")
        # Bounded smoke/debug evaluations consume the first loader batches only.
        anchors = anchors[: len(y_true)]
        payload["anchors"] = anchors
        payload["origin_time_ns"] = series.times[anchors].asi8
        payload["last_observed_target"] = series.target[anchors].astype(np.float32)
    np.savez_compressed(path, **payload)
    return str(path)


# -----------------------------------------------------------------------------
# E1: corrected in-domain benchmark
# -----------------------------------------------------------------------------

def run_e1_suite(
    data_dir: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    *,
    tasks: Sequence[str] = ("SR", "WS", "WP", "EC"),
    models: Sequence[str] = ALL_E1_MODELS,
    seeds: Sequence[int] = (0, 1, 2),
    force: bool = False,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> pd.DataFrame:
    cfg.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "E1_R2_results.csv"
    key_cols = ("protocol_id", "task", "model", "seed", "config_hash")
    existing = set() if force else load_existing_keys(results_path, key_cols)

    for task in tasks:
        series = load_task_series(task, data_dir, cfg, verbose=True)
        prepared = prepare_e1_data(series, cfg)
        print(
            f"[E1:{task}] train={len(prepared.splits['train']):,}, "
            f"val={len(prepared.splits['val']):,}, calib={len(prepared.splits['calib']):,}, "
            f"test={len(prepared.splits['test']):,}"
        )
        for model_name in models:
            if model_name not in ALL_E1_MODELS:
                raise ValueError(f"Unsupported E1 model: {model_name}")
            for seed in seeds:
                identity = {
                    **base_result_metadata(task, int(seed), cfg, prepared),
                    "model": model_name,
                }
                key = result_key(identity, key_cols)
                if key in existing:
                    print(f"[E1] skip completed {task}/{model_name}/seed={seed}")
                    continue

                set_seed(int(seed), cfg.deterministic)
                model = build_model(
                    model_name,
                    len(series.feature_names),
                    series.target_feature_index,
                    cfg,
                )
                train_loader = prepared.loader("train", cfg, int(seed), shuffle=True)
                val_loader = prepared.loader("val", cfg, int(seed), shuffle=False)
                calib_loader = prepared.loader("calib", cfg, int(seed), shuffle=False)
                test_loader = prepared.loader("test", cfg, int(seed), shuffle=False)
                lr = cfg.lr_transformer if model_name in {"patchtst", "itransformer"} else cfg.lr_full
                model, fit_stats = fit_model(
                    model,
                    train_loader,
                    val_loader,
                    cfg,
                    learning_rate=lr,
                    max_epochs=cfg.max_epochs_e1,
                    max_train_batches=max_train_batches,
                    max_val_batches=max_eval_batches,
                )
                _, calibration_true, calibration_pred, _ = evaluate_model(
                    model, calib_loader, prepared.scaler, cfg.horizons,
                    max_batches=max_eval_batches
                )
                metrics, y_true, y_pred, activity = evaluate_model(
                    model,
                    test_loader,
                    prepared.scaler,
                    cfg.horizons,
                    max_batches=max_eval_batches,
                )
                uncertainty, lower95, upper95, _ = conformal_interval_metrics(
                    calibration_true, calibration_pred, y_true, y_pred, cfg.horizons
                )
                latency_sample, latency_batch = measure_inference_latency_ms(
                    model, test_loader, max_batches=max(1, cfg.measure_latency_batches)
                )
                row: Dict[str, Any] = {
                    **identity,
                    **fit_stats_to_row(fit_stats),
                    **metrics,
                    **uncertainty,
                    **activity,
                    "inference_ms_per_sample": latency_sample,
                    "inference_ms_per_batch": latency_batch,
                    "train_windows": len(prepared.splits["train"]),
                    "val_windows": len(prepared.splits["val"]),
                    "calibration_windows": len(prepared.splits["calib"]),
                    "test_windows": len(prepared.splits["test"]),
                    "model_hparams": stable_json({
                        "dropout": cfg.dropout,
                        "transformer_d_model": cfg.transformer_d_model,
                        "transformer_heads": cfg.transformer_heads,
                        "transformer_layers": cfg.transformer_layers,
                        "patch_len": cfg.patch_len,
                        "patch_stride": cfg.patch_stride,
                        "spike_steps": cfg.spike_steps,
                        "encoder_mode": cfg.encoder_mode,
                        "delta_threshold": cfg.delta_threshold,
                    }),
                }
                if cfg.save_predictions:
                    pred_identity = f"{task}_{model_name}_seed{seed}_{cfg.protocol_id}"
                    row["prediction_file"] = save_predictions(
                        output, "E1", pred_identity, y_true, y_pred, cfg.horizons,
                        lower95=lower95, upper95=upper95,
                        anchors=prepared.splits["test"], series=prepared.series
                    )
                append_result(results_path, row, key_cols)
                existing.add(key)
                print(
                    f"[E1] {task:>2} {model_name:<12} seed={seed} "
                    f"RMSE={metrics['test_rmse']:.5g} MAE={metrics['test_mae']:.5g} "
                    f"params={fit_stats.trainable_params:,}/{fit_stats.total_params:,}"
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()


# -----------------------------------------------------------------------------
# E2: genuine disjoint temporal few-shot adaptation
# -----------------------------------------------------------------------------

def _base_checkpoint_paths(
    output: Path,
    task: str,
    seed: int,
    cfg: ExperimentConfig,
    tag: str = "E2",
) -> Tuple[Path, Path]:
    folder = output / "checkpoints" / tag
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_base_{task}_seed{seed}_{config_fingerprint(cfg)}"
    return folder / f"{stem}.pt", folder / f"{stem}.json"


def get_or_train_base_snn(
    prepared: PreparedData,
    cfg: ExperimentConfig,
    output_dir: str | Path,
    *,
    seed: int,
    task: str,
    tag: str = "E2",
    force: bool = False,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    output = Path(output_dir)
    checkpoint_path, metadata_path = _base_checkpoint_paths(output, task, seed, cfg, tag=tag)
    if checkpoint_path.exists() and metadata_path.exists() and not force:
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility
            state = torch.load(checkpoint_path, map_location="cpu")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        print(f"[{tag}] loaded base checkpoint {checkpoint_path.name}")
        return state, metadata

    set_seed(seed, cfg.deterministic)
    model = build_model(
        "snn_tcn", len(prepared.series.feature_names), prepared.series.target_feature_index, cfg
    )
    train_loader = prepared.loader("base_train", cfg, seed, shuffle=True)
    val_loader = prepared.loader("base_val", cfg, seed, shuffle=False)
    model, fit_stats = fit_model(
        model,
        train_loader,
        val_loader,
        cfg,
        learning_rate=cfg.lr_full,
        max_epochs=cfg.max_epochs_base,
        max_train_batches=max_train_batches,
        max_val_batches=max_eval_batches,
    )
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state, checkpoint_path)
    metadata = {
        **fit_stats_to_row(fit_stats),
        "protocol_id": cfg.protocol_id,
        "task": task,
        "seed": seed,
        "config_hash": config_fingerprint(cfg),
        "checkpoint": str(checkpoint_path),
        "base_train_windows": len(prepared.splits["base_train"]),
        "base_val_windows": len(prepared.splits["base_val"]),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, metadata


def build_adaptation_model(
    base_state: Mapping[str, torch.Tensor],
    prepared: PreparedData,
    cfg: ExperimentConfig,
    mode: str,
    *,
    rank: Optional[int] = None,
    gate_threshold: Optional[float] = None,
    adapter_targets: Optional[Sequence[str]] = None,
) -> nn.Module:
    if mode not in ADAPTATION_MODES:
        raise ValueError(f"Unknown adaptation mode: {mode}")
    base = build_model(
        "snn_tcn", len(prepared.series.feature_names), prepared.series.target_feature_index, cfg
    )
    assert isinstance(base, SpikingTCN)
    base.load_state_dict(dict(base_state), strict=True)
    if mode == "fullft":
        for p in base.parameters():
            p.requires_grad = True
        return base
    return SpikingTCNAdapters(
        base,
        mode=mode,
        rank=cfg.lora_rank if rank is None else int(rank),
        alpha=cfg.lora_alpha,
        gate_threshold=(cfg.spikelora_threshold if gate_threshold is None else float(gate_threshold)),
        lif_tau=cfg.lif_tau,
        targets=(cfg.adapter_targets if adapter_targets is None else tuple(adapter_targets)),
    )


def select_recent_fraction(anchors: np.ndarray, fraction: float) -> np.ndarray:
    if not (0 < fraction <= 1):
        raise ValueError("few-shot fraction must lie in (0, 1]")
    n = max(1, int(math.ceil(len(anchors) * fraction)))
    return anchors[-n:]


def run_e2_suite(
    data_dir: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    *,
    tasks: Sequence[str] = ("SR", "WS", "WP", "EC"),
    modes: Sequence[str] = ADAPTATION_MODES,
    seeds: Sequence[int] = (0, 1, 2),
    fractions: Optional[Sequence[float]] = None,
    force: bool = False,
    force_base: bool = False,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> pd.DataFrame:
    cfg.validate()
    fractions = tuple(cfg.fewshot_fractions if fractions is None else fractions)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "E2_R2_fewshot_results.csv"
    key_cols = ("protocol_id", "task", "mode", "seed", "fraction", "config_hash")
    existing = set() if force else load_existing_keys(results_path, key_cols)

    for task in tasks:
        series = load_task_series(task, data_dir, cfg, verbose=True)
        prepared = prepare_adaptation_data(series, cfg)
        print(
            f"[E2:{task}] base_train={len(prepared.splits['base_train']):,}, "
            f"base_val={len(prepared.splits['base_val']):,}, "
            f"adapt_pool={len(prepared.splits['adapt_pool']):,}, "
            f"adapt_val={len(prepared.splits['adapt_val']):,}, "
            f"calib={len(prepared.splits['calib']):,}, test={len(prepared.splits['test']):,}"
        )
        for seed in seeds:
            base_state, base_meta = get_or_train_base_snn(
                prepared,
                cfg,
                output,
                seed=int(seed),
                task=task,
                tag="E2",
                force=force_base,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
            )
            for fraction in fractions:
                selected = select_recent_fraction(prepared.splits["adapt_pool"], float(fraction))

                for mode in modes:
                    # Fresh loaders give every adaptation method the same seeded
                    # minibatch order, preserving a paired comparison.
                    train_loader = prepared.loader(
                        "adapt_pool", cfg, int(seed) + 101, shuffle=True, anchors_override=selected
                    )
                    val_loader = prepared.loader("adapt_val", cfg, int(seed) + 102, shuffle=False)
                    calib_loader = prepared.loader("calib", cfg, int(seed) + 103, shuffle=False)
                    test_loader = prepared.loader("test", cfg, int(seed) + 104, shuffle=False)
                    identity = {
                        **base_result_metadata(task, int(seed), cfg, prepared),
                        "mode": mode,
                        "fraction": float(fraction),
                    }
                    key = result_key(identity, key_cols)
                    if key in existing:
                        print(f"[E2] skip completed {task}/{mode}/seed={seed}/frac={fraction}")
                        continue

                    set_seed(int(seed) + int(round(float(fraction) * 1000)), cfg.deterministic)
                    model = build_adaptation_model(base_state, prepared, cfg, mode)
                    lr = cfg.lr_adapt_fullft if mode == "fullft" else cfg.lr_adapt_peft
                    model, fit_stats = fit_model(
                        model,
                        train_loader,
                        val_loader,
                        cfg,
                        learning_rate=lr,
                        max_epochs=cfg.max_epochs_adapt,
                        max_train_batches=max_train_batches,
                        max_val_batches=max_eval_batches,
                    )
                    _, calibration_true, calibration_pred, _ = evaluate_model(
                        model, calib_loader, prepared.scaler, cfg.horizons,
                        max_batches=max_eval_batches
                    )
                    metrics, y_true, y_pred, activity = evaluate_model(
                        model,
                        test_loader,
                        prepared.scaler,
                        cfg.horizons,
                        max_batches=max_eval_batches,
                    )
                    uncertainty, lower95, upper95, _ = conformal_interval_metrics(
                        calibration_true, calibration_pred, y_true, y_pred, cfg.horizons
                    )
                    latency_sample, latency_batch = measure_inference_latency_ms(
                        model, test_loader, max_batches=max(1, cfg.measure_latency_batches)
                    )
                    row: Dict[str, Any] = {
                        **identity,
                        **fit_stats_to_row(fit_stats),
                        **metrics,
                        **uncertainty,
                        **activity,
                        "inference_ms_per_sample": latency_sample,
                        "inference_ms_per_batch": latency_batch,
                        "selected_adaptation_windows": len(selected),
                        "adaptation_pool_windows": len(prepared.splits["adapt_pool"]),
                        "fraction_of_total_valid_windows": len(selected) / sum(
                            len(v) for v in prepared.splits.values()
                        ),
                        "base_pretrain_time_sec": base_meta.get("train_time_sec", float("nan")),
                        "base_checkpoint": base_meta.get("checkpoint", ""),
                        "lora_rank": cfg.lora_rank if mode != "fullft" else np.nan,
                        "lora_alpha": cfg.lora_alpha if mode != "fullft" else np.nan,
                        "spikelora_threshold": cfg.spikelora_threshold if mode == "spikelora" else np.nan,
                        "adapter_targets": stable_json(cfg.adapter_targets) if mode != "fullft" else "[]",
                        "strict_adapter_only": bool(mode != "fullft"),
                    }
                    if cfg.save_predictions:
                        pred_identity = f"{task}_{mode}_seed{seed}_frac{fraction}_{cfg.protocol_id}"
                        row["prediction_file"] = save_predictions(
                            output, "E2", pred_identity, y_true, y_pred, cfg.horizons,
                            lower95=lower95, upper95=upper95,
                            anchors=prepared.splits["test"], series=prepared.series
                        )
                    append_result(results_path, row, key_cols)
                    existing.add(key)
                    print(
                        f"[E2] {task:>2} {mode:<10} seed={seed} frac={fraction:.2f} "
                        f"n={len(selected):,} RMSE={metrics['test_rmse']:.5g} "
                        f"sparsity={activity.get('spikelora_sparsity_pct', float('nan')):.3g}%"
                    )
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()


# -----------------------------------------------------------------------------
# E4: targeted mechanistic ablations
# -----------------------------------------------------------------------------

def default_ablation_variants(cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    return [
        {"family": "mechanism", "label": "LoRA-r8", "mode": "lora", "rank": 8,
         "gate_threshold": np.nan, "adapter_targets": ("in_proj", "head"), "encoder_mode": "delta"},
        {"family": "mechanism", "label": "SpikeLoRA-r8-v0.05", "mode": "spikelora", "rank": 8,
         "gate_threshold": 0.05, "adapter_targets": ("in_proj", "head"), "encoder_mode": "delta"},
        {"family": "rank", "label": "SpikeLoRA-r4", "mode": "spikelora", "rank": 4,
         "gate_threshold": cfg.spikelora_threshold, "adapter_targets": ("in_proj", "head"), "encoder_mode": "delta"},
        {"family": "rank", "label": "SpikeLoRA-r16", "mode": "spikelora", "rank": 16,
         "gate_threshold": cfg.spikelora_threshold, "adapter_targets": ("in_proj", "head"), "encoder_mode": "delta"},
        {"family": "threshold", "label": "SpikeLoRA-v0.025", "mode": "spikelora", "rank": cfg.lora_rank,
         "gate_threshold": 0.025, "adapter_targets": ("in_proj", "head"), "encoder_mode": "delta"},
        {"family": "threshold", "label": "SpikeLoRA-v0.10", "mode": "spikelora", "rank": cfg.lora_rank,
         "gate_threshold": 0.10, "adapter_targets": ("in_proj", "head"), "encoder_mode": "delta"},
        {"family": "placement", "label": "SpikeLoRA-head-only", "mode": "spikelora", "rank": cfg.lora_rank,
         "gate_threshold": cfg.spikelora_threshold, "adapter_targets": ("head",), "encoder_mode": "delta"},
        {"family": "encoding", "label": "SpikeLoRA-continuous-input", "mode": "spikelora",
         "rank": cfg.lora_rank, "gate_threshold": cfg.spikelora_threshold,
         "adapter_targets": ("in_proj", "head"), "encoder_mode": "continuous"},
    ]


def run_e4_ablation_suite(
    data_dir: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    *,
    task: str = "SR",
    fraction: float = 0.10,
    seeds: Sequence[int] = (0, 1, 2),
    variants: Optional[Sequence[Mapping[str, Any]]] = None,
    force: bool = False,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> pd.DataFrame:
    variants = list(default_ablation_variants(cfg) if variants is None else variants)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "E4_R2_ablation_results.csv"
    key_cols = (
        "protocol_id", "task", "seed", "fraction", "ablation_label",
        "config_hash", "variant_config_hash"
    )
    existing = set() if force else load_existing_keys(results_path, key_cols)

    series = load_task_series(task, data_dir, cfg, verbose=True)
    prepared = prepare_adaptation_data(series, cfg)
    selected = select_recent_fraction(prepared.splits["adapt_pool"], fraction)

    for seed in seeds:
        # Cache one base per encoding condition.  The default delta-encoded base is
        # exactly the E2 checkpoint; the continuous-input ablation is pretrained
        # separately so the encoding comparison is not confounded.
        base_cache: Dict[str, Tuple[Dict[str, torch.Tensor], Dict[str, Any], ExperimentConfig]] = {}

        for variant in variants:
            train_loader = prepared.loader(
                "adapt_pool", cfg, int(seed) + 301, shuffle=True, anchors_override=selected
            )
            val_loader = prepared.loader("adapt_val", cfg, int(seed) + 302, shuffle=False)
            calib_loader = prepared.loader("calib", cfg, int(seed) + 303, shuffle=False)
            test_loader = prepared.loader("test", cfg, int(seed) + 304, shuffle=False)
            mode = str(variant["mode"])
            rank = int(variant.get("rank", cfg.lora_rank))
            raw_threshold = variant.get("gate_threshold", cfg.spikelora_threshold)
            gate_threshold = cfg.spikelora_threshold if pd.isna(raw_threshold) else float(raw_threshold)
            targets = tuple(variant.get("adapter_targets", cfg.adapter_targets))
            label = str(variant["label"])
            family = str(variant["family"])
            encoder_mode = str(variant.get("encoder_mode", cfg.encoder_mode))
            variant_cfg = replace(cfg, encoder_mode=encoder_mode)
            if encoder_mode not in base_cache:
                base_tag = "E2" if encoder_mode == cfg.encoder_mode else f"E4-{encoder_mode}"
                base_cache[encoder_mode] = (*get_or_train_base_snn(
                    prepared,
                    variant_cfg,
                    output,
                    seed=int(seed),
                    task=task,
                    tag=base_tag,
                    force=False,
                    max_train_batches=max_train_batches,
                    max_eval_batches=max_eval_batches,
                ), variant_cfg)
            base_state, base_meta, variant_cfg = base_cache[encoder_mode]

            identity = {
                **base_result_metadata(task, int(seed), cfg, prepared),
                "fraction": float(fraction),
                "ablation_label": label,
                "variant_config_hash": config_fingerprint(variant_cfg),
            }
            key = result_key(identity, key_cols)
            if key in existing:
                print(f"[E4] skip completed {label}/seed={seed}")
                continue

            set_seed(int(seed) + 400, cfg.deterministic)
            model = build_adaptation_model(
                base_state,
                prepared,
                variant_cfg,
                mode,
                rank=rank,
                gate_threshold=gate_threshold,
                adapter_targets=targets,
            )
            model, fit_stats = fit_model(
                model,
                train_loader,
                val_loader,
                variant_cfg,
                learning_rate=cfg.lr_adapt_peft,
                max_epochs=cfg.max_epochs_adapt,
                max_train_batches=max_train_batches,
                max_val_batches=max_eval_batches,
            )
            _, calibration_true, calibration_pred, _ = evaluate_model(
                model, calib_loader, prepared.scaler, cfg.horizons, max_batches=max_eval_batches
            )
            metrics, y_true, y_pred, activity = evaluate_model(
                model, test_loader, prepared.scaler, cfg.horizons, max_batches=max_eval_batches
            )
            uncertainty, lower95, upper95, _ = conformal_interval_metrics(
                calibration_true, calibration_pred, y_true, y_pred, cfg.horizons
            )
            row: Dict[str, Any] = {
                **identity,
                **fit_stats_to_row(fit_stats),
                **metrics,
                **uncertainty,
                **activity,
                "ablation_family": family,
                "mode": mode,
                "rank": rank,
                "gate_threshold": gate_threshold if mode == "spikelora" else np.nan,
                "adapter_targets": stable_json(targets),
                "encoder_mode": encoder_mode,
                "selected_adaptation_windows": len(selected),
                "base_checkpoint": base_meta.get("checkpoint", ""),
            }
            if cfg.save_predictions:
                pred_identity = f"{task}_{label}_seed{seed}_frac{fraction}_{cfg.protocol_id}"
                row["prediction_file"] = save_predictions(
                    output, "E4", pred_identity, y_true, y_pred, cfg.horizons,
                    lower95=lower95, upper95=upper95,
                    anchors=prepared.splits["test"], series=prepared.series
                )
            append_result(results_path, row, key_cols)
            existing.add(key)
            print(
                f"[E4] {label:<25} seed={seed} RMSE={metrics['test_rmse']:.5g} "
                f"sparsity={activity.get('spikelora_sparsity_pct', float('nan')):.3g}%"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()


# -----------------------------------------------------------------------------
# E3: corrected pairwise transfer with canonical feature alignment
# -----------------------------------------------------------------------------

CANONICAL_FEATURE_ORDER: Tuple[str, ...] = (
    "target_history",
    "temperature",
    "humidity",
    "dew_point",
    "cloud_type",
    "solar_zenith",
    "pressure",
    "wind_direction",
    "wind_speed",
    "cal_hour",
    "cal_dayofweek",
    "cal_month",
)

CANONICAL_MAP: Dict[str, Dict[str, str]] = {
    "SR": {
        "GHI": "target_history",
        "Temperature": "temperature",
        "Relative Humidity": "humidity",
        "Dew Point": "dew_point",
        "Cloud Type": "cloud_type",
        "Solar Zenith Angle": "solar_zenith",
        "Pressure": "pressure",
        "Wind Direction": "wind_direction",
        "Wind Speed": "wind_speed",
        "cal_hour": "cal_hour",
        "cal_dayofweek": "cal_dayofweek",
        "cal_month": "cal_month",
    },
    "WS": {
        "Wind Speed": "target_history",
        "Temperature": "temperature",
        "Relative Humidity": "humidity",
        "Dew Point": "dew_point",
        "Cloud Type": "cloud_type",
        "Pressure": "pressure",
        "Wind Direction": "wind_direction",
        "cal_hour": "cal_hour",
        "cal_dayofweek": "cal_dayofweek",
        "cal_month": "cal_month",
    },
    "WP": {
        "LV ActivePower (kW)": "target_history",
        "Wind Speed (m/s)": "wind_speed",
        "Wind Direction (°)": "wind_direction",
        "cal_hour": "cal_hour",
        "cal_dayofweek": "cal_dayofweek",
        "cal_month": "cal_month",
    },
    "EC": {
        "PowerConsumption_Zone1": "target_history",
        "Temperature": "temperature",
        "Humidity": "humidity",
        "WindSpeed": "wind_speed",
        "cal_hour": "cal_hour",
        "cal_dayofweek": "cal_dayofweek",
        "cal_month": "cal_month",
    },
}


def canonicalise_series(series: TaskSeries) -> TaskSeries:
    mapping = CANONICAL_MAP[series.task]
    name_to_index = {name: i for i, name in enumerate(series.feature_names)}
    canonical = np.zeros((len(series.times), len(CANONICAL_FEATURE_ORDER)), dtype=np.float32)
    for original, canonical_name in mapping.items():
        if original not in name_to_index:
            continue
        target_idx = CANONICAL_FEATURE_ORDER.index(canonical_name)
        canonical[:, target_idx] = series.features[:, name_to_index[original]]
    audit = dict(series.audit)
    audit["canonical_feature_alignment"] = stable_json(CANONICAL_FEATURE_ORDER)
    return TaskSeries(
        task=series.task,
        spec=series.spec,
        times=series.times,
        feature_names=CANONICAL_FEATURE_ORDER,
        features=canonical,
        target=series.target,
        input_valid=series.input_valid,
        label_observed=series.label_observed,
        target_feature_index=CANONICAL_FEATURE_ORDER.index("target_history"),
        audit=audit,
    )


def run_e3_pairwise_transfer(
    data_dir: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    *,
    source_task: str,
    target_task: str,
    seeds: Sequence[int] = (0, 1, 2),
    fractions: Optional[Sequence[float]] = None,
    modes: Sequence[str] = ADAPTATION_MODES,
    force: bool = False,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> pd.DataFrame:
    if source_task == target_task:
        raise ValueError("E3 source and target must differ")
    fractions = tuple(cfg.fewshot_fractions if fractions is None else fractions)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "E3_R2_pairwise_transfer_results.csv"
    key_cols = (
        "protocol_id", "source_task", "target_task", "mode", "seed", "fraction", "config_hash"
    )
    existing = set() if force else load_existing_keys(path, key_cols)

    source = canonicalise_series(load_task_series(source_task, data_dir, cfg, verbose=True))
    target = canonicalise_series(load_task_series(target_task, data_dir, cfg, verbose=True))
    source_prepared = prepare_adaptation_data(source, cfg)
    target_prepared = prepare_adaptation_data(target, cfg)

    for seed in seeds:
        # Source-domain base checkpoint uses a transfer-specific tag and canonical inputs.
        base_state, base_meta = get_or_train_base_snn(
            source_prepared,
            cfg,
            output,
            seed=int(seed),
            task=f"{source_task}_canonical",
            tag="E3",
            force=False,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        )
        for fraction in fractions:
            selected = select_recent_fraction(target_prepared.splits["adapt_pool"], float(fraction))
            for mode in modes:
                train_loader = target_prepared.loader(
                    "adapt_pool", cfg, int(seed) + 501, shuffle=True, anchors_override=selected
                )
                val_loader = target_prepared.loader("adapt_val", cfg, int(seed) + 502, shuffle=False)
                calib_loader = target_prepared.loader("calib", cfg, int(seed) + 503, shuffle=False)
                test_loader = target_prepared.loader("test", cfg, int(seed) + 504, shuffle=False)
                identity = {
                    **base_result_metadata(target_task, int(seed), cfg, target_prepared),
                    "source_task": source_task,
                    "target_task": target_task,
                    "mode": mode,
                    "fraction": float(fraction),
                }
                key = result_key(identity, key_cols)
                if key in existing:
                    print(f"[E3] skip {source_task}->{target_task}/{mode}/seed={seed}/frac={fraction}")
                    continue
                set_seed(int(seed) + 500, cfg.deterministic)
                model = build_adaptation_model(base_state, target_prepared, cfg, mode)
                lr = cfg.lr_adapt_fullft if mode == "fullft" else cfg.lr_adapt_peft
                model, fit_stats = fit_model(
                    model,
                    train_loader,
                    val_loader,
                    cfg,
                    learning_rate=lr,
                    max_epochs=cfg.max_epochs_adapt,
                    max_train_batches=max_train_batches,
                    max_val_batches=max_eval_batches,
                )
                _, calibration_true, calibration_pred, _ = evaluate_model(
                    model, calib_loader, target_prepared.scaler, cfg.horizons,
                    max_batches=max_eval_batches
                )
                metrics, y_true, y_pred, activity = evaluate_model(
                    model, test_loader, target_prepared.scaler, cfg.horizons,
                    max_batches=max_eval_batches
                )
                uncertainty, lower95, upper95, _ = conformal_interval_metrics(
                    calibration_true, calibration_pred, y_true, y_pred, cfg.horizons
                )
                row: Dict[str, Any] = {
                    **identity,
                    **fit_stats_to_row(fit_stats),
                    **metrics,
                    **uncertainty,
                    **activity,
                    "canonical_features": stable_json(CANONICAL_FEATURE_ORDER),
                    "selected_adaptation_windows": len(selected),
                    "source_base_checkpoint": base_meta.get("checkpoint", ""),
                    "transfer_interpretation": "boundary_analysis",
                }
                if cfg.save_predictions:
                    pred_identity = (
                        f"{source_task}_to_{target_task}_{mode}_seed{seed}_frac{fraction}_{cfg.protocol_id}"
                    )
                    row["prediction_file"] = save_predictions(
                        output, "E3", pred_identity, y_true, y_pred, cfg.horizons,
                        lower95=lower95, upper95=upper95,
                        anchors=target_prepared.splits["test"], series=target_prepared.series
                    )
                append_result(path, row, key_cols)
                existing.add(key)
                print(
                    f"[E3] {source_task}->{target_task} {mode:<10} seed={seed} frac={fraction:.2f} "
                    f"RMSE={metrics['test_rmse']:.5g}"
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# -----------------------------------------------------------------------------
# Statistical summaries and paired validation
# -----------------------------------------------------------------------------

def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = std / math.sqrt(n)
        critical = float(scipy_stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
        half = critical * sem
    else:
        half = float("nan")
    return mean, std, mean - half, mean + half, n


def summarise_results(
    frame_or_path: pd.DataFrame | str | Path,
    group_cols: Sequence[str],
    metrics: Sequence[str] = (
        "test_rmse", "test_mae", "test_smape", "picp95", "mpiw95",
        "train_time_sec", "trainable_ratio"
    ),
) -> pd.DataFrame:
    frame = pd.read_csv(frame_or_path) if isinstance(frame_or_path, (str, Path)) else frame_or_path.copy()
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in group.columns:
                continue
            mean, std, lo, hi, n = mean_std_ci(group[metric].to_numpy(dtype=float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
            row[f"{metric}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int = 2026,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        means[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_e2_comparison(
    e2_frame_or_path: pd.DataFrame | str | Path,
    method_a: str = "spikelora",
    method_b: str = "lora",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frame = pd.read_csv(e2_frame_or_path) if isinstance(e2_frame_or_path, (str, Path)) else e2_frame_or_path.copy()
    key_cols = ["task", "seed", "fraction"]
    a = frame[frame["mode"] == method_a][key_cols + ["test_rmse"]].rename(
        columns={"test_rmse": f"rmse_{method_a}"}
    )
    b = frame[frame["mode"] == method_b][key_cols + ["test_rmse"]].rename(
        columns={"test_rmse": f"rmse_{method_b}"}
    )
    paired = a.merge(b, on=key_cols, how="inner")
    paired["rmse_difference_b_minus_a"] = paired[f"rmse_{method_b}"] - paired[f"rmse_{method_a}"]
    paired["relative_improvement_pct"] = (
        100.0 * paired["rmse_difference_b_minus_a"] / paired[f"rmse_{method_b}"].replace(0, np.nan)
    )
    differences = paired["rmse_difference_b_minus_a"].to_numpy(dtype=float)
    if len(differences) and np.any(np.abs(differences) > 0):
        wilcoxon = scipy_stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
        statistic = float(wilcoxon.statistic)
        p_value = float(wilcoxon.pvalue)
    else:
        statistic = float("nan")
        p_value = float("nan")
    ci_low, ci_high = bootstrap_mean_ci(differences)
    summary = {
        "method_a": method_a,
        "method_b": method_b,
        "n_paired_conditions": int(len(paired)),
        "mean_rmse_improvement_b_minus_a": float(np.nanmean(differences)) if len(differences) else float("nan"),
        "median_rmse_improvement_b_minus_a": float(np.nanmedian(differences)) if len(differences) else float("nan"),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "win_rate_a_pct": float(100.0 * np.mean(differences > 0)) if len(differences) else float("nan"),
        "tie_rate_pct": float(100.0 * np.mean(differences == 0)) if len(differences) else float("nan"),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
    }
    return paired, summary


def build_ramp_regime_analysis(
    frame_or_path: pd.DataFrame | str | Path,
    *,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Post-hoc low/medium/high target-ramp analysis from saved predictions."""
    frame = pd.read_csv(frame_or_path) if isinstance(frame_or_path, (str, Path)) else frame_or_path.copy()
    if "prediction_file" not in frame.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    identity_cols = [
        c for c in (
            "protocol_id", "task", "model", "mode", "source_task", "target_task",
            "seed", "fraction", "ablation_family", "ablation_label"
        ) if c in frame.columns
    ]
    for record in frame.to_dict(orient="records"):
        prediction_file = record.get("prediction_file")
        if not isinstance(prediction_file, str) or not prediction_file or not Path(prediction_file).exists():
            continue
        with np.load(prediction_file) as data:
            required = {"y_true", "y_pred", "last_observed_target", "horizons"}
            if not required.issubset(set(data.files)):
                continue
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            last = data["last_observed_target"].reshape(-1, 1)
            horizons = tuple(int(v) for v in data["horizons"].tolist())
            ramp = np.max(np.abs(y_true - last), axis=1)
            q33, q67 = np.quantile(ramp, [1.0 / 3.0, 2.0 / 3.0])
            labels = np.where(ramp <= q33, "low", np.where(ramp <= q67, "medium", "high"))
            lower = data["lower95"] if "lower95" in data.files else None
            upper = data["upper95"] if "upper95" in data.files else None
            for regime in ("low", "medium", "high"):
                mask = labels == regime
                if not np.any(mask):
                    continue
                result: Dict[str, Any] = {c: record.get(c) for c in identity_cols}
                result.update({
                    "regime": regime,
                    "regime_n": int(mask.sum()),
                    "ramp_q33": float(q33),
                    "ramp_q67": float(q67),
                    **metric_dict(y_true[mask], y_pred[mask], horizons),
                })
                if lower is not None and upper is not None:
                    covered = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
                    result["picp95"] = float(np.mean(covered))
                    result["mpiw95"] = float(np.mean(upper[mask] - lower[mask]))
                rows.append(result)
    result_frame = pd.DataFrame(rows)
    if output_path is not None and not result_frame.empty:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(out, index=False)
    return result_frame


def write_all_summaries(output_dir: str | Path) -> Dict[str, str]:
    output = Path(output_dir)
    written: Dict[str, str] = {}
    e1_path = output / "E1_R2_results.csv"
    if e1_path.exists():
        summary = summarise_results(e1_path, ("task", "model"))
        path = output / "E1_R2_summary_ci95.csv"
        summary.to_csv(path, index=False)
        written["E1_summary"] = str(path)
        regime_path = output / "E1_R2_ramp_regime_analysis.csv"
        regime = build_ramp_regime_analysis(e1_path, output_path=regime_path)
        if not regime.empty:
            regime_summary = summarise_results(regime, ("task", "model", "regime"))
            regime_summary_path = output / "E1_R2_ramp_regime_summary_ci95.csv"
            regime_summary.to_csv(regime_summary_path, index=False)
            written["E1_regime"] = str(regime_path)
            written["E1_regime_summary"] = str(regime_summary_path)
    e2_path = output / "E2_R2_fewshot_results.csv"
    if e2_path.exists():
        summary = summarise_results(e2_path, ("task", "mode", "fraction"))
        path = output / "E2_R2_summary_ci95.csv"
        summary.to_csv(path, index=False)
        paired, statistical = paired_e2_comparison(e2_path)
        paired_path = output / "E2_R2_SpikeLoRA_vs_LoRA_paired.csv"
        paired.to_csv(paired_path, index=False)
        stats_path = output / "E2_R2_SpikeLoRA_vs_LoRA_statistics.json"
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(statistical, handle, indent=2, sort_keys=True)
        written["E2_summary"] = str(path)
        written["E2_paired"] = str(paired_path)
        written["E2_statistics"] = str(stats_path)
        regime_path = output / "E2_R2_ramp_regime_analysis.csv"
        regime = build_ramp_regime_analysis(e2_path, output_path=regime_path)
        if not regime.empty:
            regime_summary = summarise_results(regime, ("task", "mode", "fraction", "regime"))
            regime_summary_path = output / "E2_R2_ramp_regime_summary_ci95.csv"
            regime_summary.to_csv(regime_summary_path, index=False)
            written["E2_regime"] = str(regime_path)
            written["E2_regime_summary"] = str(regime_summary_path)
    e4_path = output / "E4_R2_ablation_results.csv"
    if e4_path.exists():
        summary = summarise_results(e4_path, ("task", "ablation_family", "ablation_label", "fraction"))
        path = output / "E4_R2_ablation_summary_ci95.csv"
        summary.to_csv(path, index=False)
        written["E4_summary"] = str(path)
    e3_path = output / "E3_R2_pairwise_transfer_results.csv"
    if e3_path.exists():
        summary = summarise_results(e3_path, ("source_task", "target_task", "mode", "fraction"))
        path = output / "E3_R2_pairwise_summary_ci95.csv"
        summary.to_csv(path, index=False)
        written["E3_summary"] = str(path)
    return written


# -----------------------------------------------------------------------------
# Smoke tests
# -----------------------------------------------------------------------------

def smoke_test(
    data_dir: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    *,
    tasks: Sequence[str] = ("SR", "EC"),
    models: Sequence[str] = ("patchtst", "itransformer", "snn_tcn"),
) -> pd.DataFrame:
    """Fast shape/training test using bounded batches; does not replace full runs."""
    smoke_cfg = replace(
        cfg,
        batch_size=min(32, cfg.batch_size),
        max_epochs_e1=1,
        spike_steps=min(2, cfg.spike_steps),
        snn_channels=(16, 16, 16),
        transformer_d_model=32,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff=64,
        measure_latency_batches=1,
        deterministic=False,
        save_predictions=False,
    )
    smoke_dir = Path(output_dir) / "smoke"
    frame = run_e1_suite(
        data_dir,
        smoke_dir,
        smoke_cfg,
        tasks=tasks,
        models=models,
        seeds=(0,),
        force=True,
        max_train_batches=1,
        max_eval_batches=1,
    )
    if frame.empty or not np.isfinite(frame["test_rmse"]).all():
        raise RuntimeError("Smoke test failed: non-finite or absent results")
    print("Smoke test passed for:", frame[["task", "model", "test_rmse"]].to_dict("records"))
    return frame


# === CLI ENTRYPOINT ===

def _parse_csv_list(value: str, cast=str) -> Tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "smoke", "e1", "e2", "e4", "e3", "summaries"))
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="./SpikeLoRA_R2_results")
    parser.add_argument("--tasks", default="SR,WS,WP,EC")
    parser.add_argument("--models", default=",".join(ALL_E1_MODELS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--fractions", default="0.1,0.2,0.5")
    parser.add_argument("--source", default="WS")
    parser.add_argument("--target", default="WP")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig()
    tasks = _parse_csv_list(args.tasks, str)
    seeds = _parse_csv_list(args.seeds, int)
    fractions = _parse_csv_list(args.fractions, float)

    if args.command == "audit":
        print(audit_all_datasets(args.data_dir, args.output_dir, cfg, tasks=tasks).to_string(index=False))
    elif args.command == "smoke":
        smoke_test(args.data_dir, args.output_dir, cfg)
    elif args.command == "e1":
        run_e1_suite(
            args.data_dir, args.output_dir, cfg, tasks=tasks,
            models=_parse_csv_list(args.models, str), seeds=seeds, force=args.force
        )
    elif args.command == "e2":
        run_e2_suite(
            args.data_dir, args.output_dir, cfg, tasks=tasks, seeds=seeds,
            fractions=fractions, force=args.force
        )
    elif args.command == "e4":
        run_e4_ablation_suite(
            args.data_dir, args.output_dir, cfg, task=tasks[0],
            fraction=float(fractions[0]), seeds=seeds, force=args.force
        )
    elif args.command == "e3":
        run_e3_pairwise_transfer(
            args.data_dir, args.output_dir, cfg,
            source_task=args.source, target_task=args.target,
            seeds=seeds, fractions=fractions, force=args.force
        )
    elif args.command == "summaries":
        print(json.dumps(write_all_summaries(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
