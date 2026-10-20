from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


KEY_COLUMNS = ["arrival_rate", "max_batch_size", "kv_budget"]
TAIL_METRICS = ["p95_ttft", "p99_ttft"]
ALL_METRICS = [
    "mean_ttft",
    "p95_ttft",
    "p99_ttft",
    "goodput",
    "blocking_probability",
]
METHODS = ["closed_form", "markov_lite", "calibrated_tail"]
EPSILON = 1e-12


@dataclass(frozen=True)
class TailCalibrationModel:
    metric: str
    feature_names: list[str]
    train_mean: np.ndarray
    train_std: np.ndarray
    coefficients: np.ndarray


def deterministic_split(df: pd.DataFrame) -> pd.DataFrame:
    """Sort configs and reserve every fourth row for honest test evaluation."""
    split_df = df.sort_values(KEY_COLUMNS).reset_index(drop=True).copy()
    split_df["split"] = np.where(split_df.index % 4 == 0, "test", "train")
    return split_df


def relative_error(predicted: pd.Series, observed: pd.Series) -> pd.Series:
    denominator = observed.abs()
    return (predicted - observed).abs().where(
        denominator > EPSILON, 0.0
    ) / denominator.where(denominator > EPSILON, 1.0)


def _raw_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Configuration-level features used to calibrate tail residuals.

    This is intentionally lightweight: the Markov-lite model remains the
    analytical backbone, and the regression only learns a smooth multiplicative
    correction for p95/p99 tail bias.
    """
    max_batch = df["max_batch_size"].clip(lower=1)
    kv_budget = df["kv_budget"].clip(lower=1)
    markov_goodput = df["goodput_markov"].clip(lower=EPSILON)

    return pd.DataFrame(
        {
            "arrival_rate": df["arrival_rate"],
            "max_batch_size": df["max_batch_size"],
            "kv_budget": df["kv_budget"],
            "blocking_probability_markov": df["blocking_probability_markov"],
            "goodput_markov": df["goodput_markov"],
            "offered_load_proxy": df["arrival_rate"] / markov_goodput,
            "arrival_rate_per_batch": df["arrival_rate"] / max_batch,
            "arrival_rate_per_kv": df["arrival_rate"] / kv_budget,
        }
    )


def _standardize_features(
    raw_features: pd.DataFrame,
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> np.ndarray:
    standardized = (raw_features.to_numpy(dtype=float) - train_mean) / train_std
    return np.column_stack([np.ones(len(raw_features)), standardized])


def fit_tail_model(df: pd.DataFrame, metric: str) -> TailCalibrationModel:
    if metric not in TAIL_METRICS:
        raise ValueError(f"metric must be one of {TAIL_METRICS}, got {metric!r}")

    train = df[df["split"] == "train"].copy()
    if train.empty:
        raise ValueError("training split is empty")

    raw_train = _raw_feature_frame(train)
    train_mean = raw_train.to_numpy(dtype=float).mean(axis=0)
    train_std = raw_train.to_numpy(dtype=float).std(axis=0)
    train_std = np.where(train_std < EPSILON, 1.0, train_std)

    x_train = _standardize_features(raw_train, train_mean, train_std)
    sim = train[f"{metric}_sim"].clip(lower=EPSILON).to_numpy(dtype=float)
    markov = train[f"{metric}_markov"].clip(lower=EPSILON).to_numpy(dtype=float)
    y_train = np.log(sim / markov)

    coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return TailCalibrationModel(
        metric=metric,
        feature_names=["intercept"] + list(raw_train.columns),
        train_mean=train_mean,
        train_std=train_std,
        coefficients=coefficients,
    )


def predict_log_residual(df: pd.DataFrame, model: TailCalibrationModel) -> np.ndarray:
    raw_features = _raw_feature_frame(df)
    x = _standardize_features(raw_features, model.train_mean, model.train_std)
    return x @ model.coefficients


def apply_tail_calibration(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, TailCalibrationModel]]:
    """Fit p95/p99 residual models and return calibrated predictions."""
    predictions = deterministic_split(df)
    models: dict[str, TailCalibrationModel] = {}

    for metric in ALL_METRICS:
        predictions[f"{metric}_markov_lite"] = predictions[f"{metric}_markov"]

    for metric in TAIL_METRICS:
        model = fit_tail_model(predictions, metric)
        models[metric] = model
        log_residual = predict_log_residual(predictions, model)
        predictions[f"{metric}_calibrated_tail"] = (
            predictions[f"{metric}_markov"].clip(lower=EPSILON) * np.exp(log_residual)
        )
        predictions[f"{metric}_calibrated_log_residual"] = log_residual

    # The calibrated-tail method only changes p95/p99. Other metrics are copied
    # through from Markov-lite so downstream tables can use one consistent suffix.
    for metric in ALL_METRICS:
        calibrated_column = f"{metric}_calibrated_tail"
        if calibrated_column not in predictions:
            predictions[calibrated_column] = predictions[f"{metric}_markov"]

    return predictions, models


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["regime"] = np.where(
        output["blocking_probability_sim"] < 0.05, "stable", "overloaded"
    )
    for method in METHODS:
        for metric in ALL_METRICS:
            output[f"{metric}_{method}_relative_error"] = relative_error(
                output[f"{metric}_{method}"], output[f"{metric}_sim"]
            )
    return output


def overall_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ALL_METRICS:
        for method in METHODS:
            errors = predictions[f"{metric}_{method}_relative_error"]
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "mean_relative_error": errors.mean(),
                    "median_relative_error": errors.median(),
                    "calibrated_tail_changes_metric": metric in TAIL_METRICS,
                }
            )
    return pd.DataFrame(rows)


def train_test_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TAIL_METRICS:
        for method in METHODS:
            for split in ["train", "test"]:
                errors = predictions.loc[
                    predictions["split"] == split,
                    f"{metric}_{method}_relative_error",
                ]
                rows.append(
                    {
                        "metric": metric,
                        "method": method,
                        "split": split,
                        "mean_relative_error": errors.mean(),
                        "median_relative_error": errors.median(),
                    }
                )
    return pd.DataFrame(rows)


def regime_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TAIL_METRICS:
        for method in METHODS:
            for split in ["train", "test"]:
                for regime in ["stable", "overloaded"]:
                    mask = (predictions["split"] == split) & (
                        predictions["regime"] == regime
                    )
                    errors = predictions.loc[
                        mask,
                        f"{metric}_{method}_relative_error",
                    ]
                    rows.append(
                        {
                            "metric": metric,
                            "method": method,
                            "split": split,
                            "regime": regime,
                            "n_configs": int(mask.sum()),
                            "mean_relative_error": errors.mean(),
                            "median_relative_error": errors.median(),
                        }
                    )
    return pd.DataFrame(rows)
