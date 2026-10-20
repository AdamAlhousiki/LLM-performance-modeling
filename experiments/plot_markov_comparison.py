from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "mean_ttft": {
        "label": "Mean TTFT (s)",
        "title": "Simulation vs Markov-lite Mean TTFT",
        "output": "markov_comparison_mean_ttft.png",
    },
    "p99_ttft": {
        "label": "p99 TTFT (s)",
        "title": "Simulation vs Markov-lite p99 TTFT",
        "output": "markov_comparison_p99_ttft.png",
    },
    "goodput": {
        "label": "Goodput (completed requests/s)",
        "title": "Simulation vs Markov-lite Goodput",
        "output": "markov_comparison_goodput.png",
    },
    "blocking_probability": {
        "label": "Blocking Probability",
        "title": "Simulation vs Markov-lite Blocking Probability",
        "output": "markov_comparison_blocking_probability.png",
    },
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
        description="Plot SimPy vs Markov-chain-lite comparison results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "markov_comparison_summary.csv",
    )
    parser.add_argument(
        "--combined-table",
        type=Path,
        default=Path("results") / "analytical_error_table.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def plot_scatter(df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    config = METRICS[metric]
    x = df[f"{metric}_sim"]
    y = df[f"{metric}_markov"]
    upper = max(float(x.max()), float(y.max()), 1e-9)
    axis_max = upper * 1.05

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.75, edgecolors="none")
    plt.plot(
        [0.0, axis_max],
        [0.0, axis_max],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="y=x",
    )
    plt.xlim(0.0, axis_max)
    plt.ylim(0.0, axis_max)
    plt.xlabel(f"Simulation {config['label']}")
    plt.ylabel(f"Markov-lite {config['label']}")
    plt.title(config["title"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / config["output"]
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_relative_errors(table: pd.DataFrame, output_dir: Path) -> Path:
    required = {
        "metric",
        "closed_form_mean_relative_error",
        "markov_lite_mean_relative_error",
    }
    if not required.issubset(table.columns):
        raise ValueError(
            "Combined table must contain closed-form and markov-lite mean error columns."
        )

    labels = table["metric"].str.replace("_", " ", regex=False)
    x = range(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(
        [idx - width / 2 for idx in x],
        table["closed_form_mean_relative_error"],
        width=width,
        label="Closed-form",
        color="#4C78A8",
    )
    plt.bar(
        [idx + width / 2 for idx in x],
        table["markov_lite_mean_relative_error"],
        width=width,
        label="Markov-lite",
        color="#F58518",
    )
    plt.ylabel("Mean Relative Error")
    plt.title("Closed-form vs Markov-lite Relative Errors")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "markov_comparison_relative_errors.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} missing. Run `python -m experiments.compare_simulation_markov` first."
        )
    if not args.combined_table.exists():
        raise FileNotFoundError(
            f"{args.combined_table} missing. Run `python -m experiments.compare_simulation_markov` first."
        )

    df = pd.read_csv(args.input)
    table = pd.read_csv(args.combined_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        print(f"wrote {plot_scatter(df, metric, args.output_dir)}")
    print(f"wrote {plot_relative_errors(table, args.output_dir)}")


if __name__ == "__main__":
    main()
