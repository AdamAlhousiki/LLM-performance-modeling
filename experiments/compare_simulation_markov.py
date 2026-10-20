from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["arrival_rate", "max_batch_size", "kv_budget"]
METRIC_COLUMNS = [
    "mean_ttft",
    "p95_ttft",
    "p99_ttft",
    "goodput",
    "blocking_probability",
]
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
        description="Compare SimPy sweep results with Markov-chain-lite."
    )
    parser.add_argument(
        "--simulation",
        type=Path,
        default=Path("results") / "sweep_summary.csv",
    )
    parser.add_argument(
        "--markov",
        type=Path,
        default=Path("results") / "markov_summary.csv",
    )
    parser.add_argument(
        "--closed-form",
        type=Path,
        default=Path("results") / "closed_form_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "markov_comparison_summary.csv",
    )
    parser.add_argument(
        "--combined-table",
        type=Path,
        default=Path("results") / "analytical_error_table.csv",
    )
    return parser.parse_args()


def relative_error(estimated: pd.Series, observed: pd.Series) -> pd.Series:
    denominator = observed.abs()
    return (estimated - observed).abs().where(denominator > 0, 0.0) / denominator.where(
        denominator > 0, 1.0
    )


def load_simulation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python -m experiments.run_sweep` first."
        )
    sim = pd.read_csv(path)
    required = KEY_COLUMNS + [f"{metric}_mean" for metric in METRIC_COLUMNS]
    missing = [column for column in required if column not in sim]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return sim[required].rename(
        columns={f"{metric}_mean": f"{metric}_sim" for metric in METRIC_COLUMNS}
    )


def load_model(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing.")
    df = pd.read_csv(path)
    missing = [column for column in SUMMARY_COLUMNS if column not in df]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[KEY_COLUMNS + METRIC_COLUMNS].rename(
        columns={metric: f"{metric}_{model_name}" for metric in METRIC_COLUMNS}
    )


def comparison(sim: pd.DataFrame, model: pd.DataFrame, model_name: str) -> pd.DataFrame:
    merged = sim.merge(model, on=KEY_COLUMNS, how="inner")
    for metric in METRIC_COLUMNS:
        merged[f"{metric}_relative_error"] = relative_error(
            merged[f"{metric}_{model_name}"], merged[f"{metric}_sim"]
        )
    return merged


def error_summary(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for metric in METRIC_COLUMNS:
        errors = df[f"{metric}_relative_error"]
        rows.append(
            {
                "metric": metric,
                f"{model_name}_mean_relative_error": errors.mean(),
                f"{model_name}_median_relative_error": errors.median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    sim = load_simulation(args.simulation)
    markov = load_model(args.markov, "markov")
    markov_cmp = comparison(sim, markov, "markov")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    markov_cmp.to_csv(args.output, index=False)

    print(f"wrote {args.output}")
    print(f"matched configs: {len(markov_cmp)}")
    for metric in METRIC_COLUMNS:
        print(
            f"avg relative error markov-lite {metric}: "
            f"{markov_cmp[f'{metric}_relative_error'].mean():.4f}"
        )

    markov_summary = error_summary(markov_cmp, "markov_lite")
    if args.closed_form.exists():
        closed_form = load_model(args.closed_form, "closed_form")
        closed_cmp = comparison(sim, closed_form, "closed_form")
        closed_summary = error_summary(closed_cmp, "closed_form")
        combined = closed_summary.merge(markov_summary, on="metric")
        combined = combined[
            [
                "metric",
                "closed_form_mean_relative_error",
                "markov_lite_mean_relative_error",
                "closed_form_median_relative_error",
                "markov_lite_median_relative_error",
            ]
        ]
    else:
        print(
            f"Missing {args.closed_form}; run `python -m analytical.run_closed_form` "
            "for the closed-form side of the combined table."
        )
        combined = markov_summary

    combined.to_csv(args.combined_table, index=False)
    print(f"wrote {args.combined_table}")
    print()
    print("Metric | Closed-form mean relative error | Markov-lite mean relative error | Closed-form median relative error | Markov-lite median relative error")
    print("---|---:|---:|---:|---:")
    for row in combined.to_dict(orient="records"):
        print(
            f"{row['metric']} | "
            f"{row.get('closed_form_mean_relative_error', 0.0):.4f} | "
            f"{row.get('markov_lite_mean_relative_error', 0.0):.4f} | "
            f"{row.get('closed_form_median_relative_error', 0.0):.4f} | "
            f"{row.get('markov_lite_median_relative_error', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
