from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot vLLM stress sweep results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "vllm_stress_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def _series_label(row: pd.Series) -> str:
    return f"B={int(row['max_batch_size'])}, util={float(row['gpu_memory_utilization']):.2f}"


def plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    plt.figure(figsize=(8, 5))
    for (_batch, _util), group in df.groupby(
        ["max_batch_size", "gpu_memory_utilization"], sort=True
    ):
        group = group.sort_values("arrival_rate")
        plt.plot(
            group["arrival_rate"],
            group[metric],
            marker="o",
            linewidth=1.8,
            label=_series_label(group.iloc[0]),
        )
    plt.xlabel("Arrival Rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} missing. Run `python -m experiments.run_vllm_stress_sweep` "
            "or `modal run experiments/run_vllm_stress_sweep.py` first."
        )

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"wrote {plot_metric(df, metric='p99_ttft', ylabel='p99 TTFT (s)', title='vLLM Stress p99 TTFT', output_path=args.output_dir / 'vllm_stress_ttft.png')}"
    )
    print(
        f"wrote {plot_metric(df, metric='mean_tpot', ylabel='Mean TPOT (s/token)', title='vLLM Stress Mean TPOT', output_path=args.output_dir / 'vllm_stress_tpot.png')}"
    )
    print(
        f"wrote {plot_metric(df, metric='goodput', ylabel='Goodput (req/s)', title='vLLM Stress Goodput', output_path=args.output_dir / 'vllm_stress_goodput.png')}"
    )


if __name__ == "__main__":
    main()
