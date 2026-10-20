from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "closed_form": "Closed-form",
    "markov_lite": "Markov-lite",
    "calibrated_tail": "Calibrated-tail",
}
METHOD_COLORS = {
    "closed_form": "#4C78A8",
    "markov_lite": "#F58518",
    "calibrated_tail": "#54A24B",
}
TAIL_LABELS = {
    "p95_ttft": "p95 TTFT (s)",
    "p99_ttft": "p99 TTFT (s)",
}

plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot calibrated Markov-tail comparison results."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results") / "calibrated_tail_predictions.csv",
    )
    parser.add_argument(
        "--error-table",
        type=Path,
        default=Path("results") / "calibrated_tail_error_table.csv",
    )
    parser.add_argument(
        "--train-test-table",
        type=Path,
        default=Path("results") / "calibrated_tail_train_test_error_table.csv",
    )
    parser.add_argument(
        "--regime-table",
        type=Path,
        default=Path("results") / "calibrated_tail_error_by_regime.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def _axis_limit(*series: pd.Series) -> float:
    upper = max(float(values.max()) for values in series)
    return max(upper * 1.05, 1e-9)


def plot_tail_scatter(df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    sim = df[f"{metric}_sim"]
    predictions = [df[f"{metric}_{method}"] for method in METHOD_LABELS]
    axis_max = _axis_limit(sim, *predictions)

    plt.figure(figsize=(7, 6))
    for method, label in METHOD_LABELS.items():
        plt.scatter(
            sim,
            df[f"{metric}_{method}"],
            alpha=0.72,
            s=42,
            label=label,
            color=METHOD_COLORS[method],
            edgecolors="none",
        )
    plt.plot(
        [0.0, axis_max],
        [0.0, axis_max],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="y=x",
    )
    plt.xlim(0.0, axis_max)
    plt.ylim(0.0, axis_max)
    plt.xlabel(f"Simulation {TAIL_LABELS[metric]}")
    plt.ylabel(f"Predicted {TAIL_LABELS[metric]}")
    plt.title(f"Simulation vs Analytical {TAIL_LABELS[metric]}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"calibrated_tail_{metric.split('_')[0]}_scatter.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_relative_errors(error_table: pd.DataFrame, output_dir: Path) -> Path:
    tail_errors = error_table[error_table["metric"].isin(TAIL_LABELS)].copy()
    metrics = list(TAIL_LABELS)
    methods = list(METHOD_LABELS)
    width = 0.24
    x = range(len(metrics))

    plt.figure(figsize=(8, 5))
    for offset, method in enumerate(methods):
        values = []
        for metric in metrics:
            match = tail_errors[
                (tail_errors["metric"] == metric) & (tail_errors["method"] == method)
            ]
            values.append(float(match["mean_relative_error"].iloc[0]))
        positions = [idx + (offset - 1) * width for idx in x]
        plt.bar(
            positions,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )

    plt.ylabel("Mean Relative Error")
    plt.title("Tail TTFT Relative Errors")
    plt.xticks(list(x), [TAIL_LABELS[metric].replace(" (s)", "") for metric in metrics])
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "calibrated_tail_relative_errors.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_train_test_errors(split_table: pd.DataFrame, output_dir: Path) -> Path:
    metrics = list(TAIL_LABELS)
    methods = list(METHOD_LABELS)
    splits = ["train", "test"]
    labels = [
        f"{TAIL_LABELS[metric].replace(' (s)', '')}\n{split.title()}"
        for metric in metrics
        for split in splits
    ]
    x = range(len(labels))
    width = 0.24

    plt.figure(figsize=(10, 5))
    for offset, method in enumerate(methods):
        values = []
        for metric in metrics:
            for split in splits:
                match = split_table[
                    (split_table["metric"] == metric)
                    & (split_table["method"] == method)
                    & (split_table["split"] == split)
                ]
                values.append(float(match["mean_relative_error"].iloc[0]))
        positions = [idx + (offset - 1) * width for idx in x]
        plt.bar(
            positions,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )

    plt.ylabel("Mean Relative Error")
    plt.title("Tail TTFT Train/Test Relative Errors")
    plt.xticks(list(x), labels)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "calibrated_tail_train_test_errors.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_regime_errors(regime_table: pd.DataFrame, output_dir: Path) -> Path:
    filtered = regime_table[
        (regime_table["method"] == "calibrated_tail")
        & (regime_table["metric"].isin(TAIL_LABELS))
    ].copy()
    metrics = list(TAIL_LABELS)
    splits = ["train", "test"]
    regimes = ["stable", "overloaded"]
    labels = [
        f"{TAIL_LABELS[metric].replace(' (s)', '')}\n{split.title()}"
        for metric in metrics
        for split in splits
    ]
    x = range(len(labels))
    width = 0.34
    colors = {"stable": "#4C78A8", "overloaded": "#E45756"}

    plt.figure(figsize=(10, 5))
    for offset, regime in enumerate(regimes):
        values = []
        for metric in metrics:
            for split in splits:
                match = filtered[
                    (filtered["metric"] == metric)
                    & (filtered["split"] == split)
                    & (filtered["regime"] == regime)
                ]
                value = float(match["mean_relative_error"].iloc[0])
                values.append(0.0 if pd.isna(value) else value)
        positions = [idx + (offset - 0.5) * width for idx in x]
        plt.bar(
            positions,
            values,
            width=width,
            label=regime.title(),
            color=colors[regime],
        )

    plt.ylabel("Mean Relative Error")
    plt.title("Calibrated Tail Error by Regime")
    plt.xticks(list(x), labels)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Simulator Regime")
    plt.tight_layout()

    output_path = output_dir / "calibrated_tail_error_by_regime.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    for path in [
        args.predictions,
        args.error_table,
        args.train_test_table,
        args.regime_table,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run `python -m experiments.run_calibrated_tail` first."
            )

    df = pd.read_csv(args.predictions)
    error_table = pd.read_csv(args.error_table)
    split_table = pd.read_csv(args.train_test_table)
    regime_table = pd.read_csv(args.regime_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"wrote {plot_tail_scatter(df, 'p95_ttft', args.output_dir)}")
    print(f"wrote {plot_tail_scatter(df, 'p99_ttft', args.output_dir)}")
    print(f"wrote {plot_relative_errors(error_table, args.output_dir)}")
    print(f"wrote {plot_train_test_errors(split_table, args.output_dir)}")
    print(f"wrote {plot_regime_errors(regime_table, args.output_dir)}")


if __name__ == "__main__":
    main()
