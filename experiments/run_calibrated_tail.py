from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analytical.calibrated_tail import (
    ALL_METRICS,
    KEY_COLUMNS,
    add_error_columns,
    apply_tail_calibration,
    overall_error_table,
    regime_error_table,
    train_test_error_table,
)


SUMMARY_COLUMNS = [
    "arrival_rate",
    "max_batch_size",
    "kv_budget",
    "mean_ttft",
    "p50_ttft",
    "p95_ttft",
    "p99_ttft",
    "goodput",
    "blocking_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a lightweight calibrated Markov-tail correction."
    )
    parser.add_argument(
        "--markov-comparison",
        type=Path,
        default=Path("results") / "markov_comparison_summary.csv",
    )
    parser.add_argument(
        "--closed-form",
        type=Path,
        default=Path("results") / "closed_form_summary.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("results") / "calibrated_tail_predictions.csv",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=Path("results") / "calibrated_tail_error_table.csv",
    )
    parser.add_argument(
        "--train-test-output",
        type=Path,
        default=Path("results") / "calibrated_tail_train_test_error_table.csv",
    )
    parser.add_argument(
        "--regime-output",
        type=Path,
        default=Path("results") / "calibrated_tail_error_by_regime.csv",
    )
    return parser.parse_args()


def load_closed_form(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python -m analytical.run_closed_form` first."
        )
    df = pd.read_csv(path)
    missing = [column for column in SUMMARY_COLUMNS if column not in df]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[KEY_COLUMNS + ALL_METRICS].rename(
        columns={metric: f"{metric}_closed_form" for metric in ALL_METRICS}
    )


def load_markov_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python -m experiments.compare_simulation_markov` first."
        )
    df = pd.read_csv(path)
    required = list(KEY_COLUMNS)
    required += [f"{metric}_sim" for metric in ALL_METRICS]
    required += [f"{metric}_markov" for metric in ALL_METRICS]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[required]


def main() -> None:
    args = parse_args()
    markov = load_markov_comparison(args.markov_comparison)
    closed_form = load_closed_form(args.closed_form)
    merged = markov.merge(closed_form, on=KEY_COLUMNS, how="inner")
    if merged.empty:
        raise ValueError("No shared configurations between Markov and closed-form files.")

    predictions, models = apply_tail_calibration(merged)
    predictions = add_error_columns(predictions)
    error_table = overall_error_table(predictions)
    split_table = train_test_error_table(predictions)
    regime_table = regime_error_table(predictions)

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    error_table.to_csv(args.error_output, index=False)
    split_table.to_csv(args.train_test_output, index=False)
    regime_table.to_csv(args.regime_output, index=False)

    print(f"wrote {args.predictions_output}")
    print(f"wrote {args.error_output}")
    print(f"wrote {args.train_test_output}")
    print(f"wrote {args.regime_output}")
    print(f"matched configs: {len(predictions)}")
    print("tail calibration coefficients:")
    for metric, model in models.items():
        terms = ", ".join(
            f"{name}={coef:.4g}"
            for name, coef in zip(model.feature_names, model.coefficients)
        )
        print(f"  {metric}: {terms}")

    print()
    print("Overall mean/median relative errors:")
    for row in error_table.to_dict(orient="records"):
        print(
            f"  {row['method']} {row['metric']}: "
            f"mean={row['mean_relative_error']:.4f}, "
            f"median={row['median_relative_error']:.4f}"
        )

    print()
    print("Tail train/test relative errors:")
    for row in split_table.to_dict(orient="records"):
        print(
            f"  {row['method']} {row['metric']} {row['split']}: "
            f"mean={row['mean_relative_error']:.4f}, "
            f"median={row['median_relative_error']:.4f}"
        )

    print()
    print("Tail train/test relative errors by regime:")
    for row in regime_table.to_dict(orient="records"):
        mean_error = row["mean_relative_error"]
        median_error = row["median_relative_error"]
        mean_text = "nan" if pd.isna(mean_error) else f"{mean_error:.4f}"
        median_text = "nan" if pd.isna(median_error) else f"{median_error:.4f}"
        print(
            f"  {row['method']} {row['metric']} {row['split']} {row['regime']} "
            f"(n={row['n_configs']}): mean={mean_text}, median={median_text}"
        )


if __name__ == "__main__":
    main()
