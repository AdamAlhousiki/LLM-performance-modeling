from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments import run_vllm_sweep as base


DEFAULT_STRESS_ARRIVAL_RATES = "12,24"
DEFAULT_STRESS_BATCH_SIZES = "64"
DEFAULT_STRESS_GPU_MEMORY_UTILS = "0.30"
DEFAULT_STRESS_MAX_MODEL_LEN = 4096
DEFAULT_STRESS_NUM_PROMPTS = 200
DEFAULT_STRESS_NUM_WARMUPS = 4

STRESS_COLUMNS = [
    "arrival_rate",
    "max_batch_size",
    "gpu_memory_utilization",
    "max_model_len",
    "kv_budget",
    "realized_kv_cache_blocks",
    "kv_block_size",
    "num_prompts",
    "num_warmups",
    "mean_ttft",
    "p50_ttft",
    "p95_ttft",
    "p99_ttft",
    "mean_tpot",
    "p99_tpot",
    "goodput",
    "request_throughput",
    "request_goodput_slo_filtered",
    "blocking_probability",
    "preemption_count",
    "request_failures",
    "completed",
    "max_concurrency",
    "max_active_sequences",
    "output_throughput",
    "total_token_throughput",
    "vllm_version",
]


app = base.app


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_stress_summary(
    *,
    raw_summary_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> None:
    if not raw_summary_path.exists():
        raise FileNotFoundError(f"{raw_summary_path} missing")
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} missing")

    with raw_summary_path.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))
    metadata = _read_json(metadata_path)

    metadata_by_key: dict[tuple[int, float, float], dict[str, Any]] = {}
    for config in metadata.get("configs", []):
        kv_cache = config.get("kv_cache", {}) or {}
        for bench in config.get("bench_runs", []):
            key = (
                int(config["max_num_seqs"]),
                float(config["gpu_memory_utilization"]),
                float(bench["arrival_rate"]),
            )
            metadata_by_key[key] = {
                "config": config,
                "bench": bench,
                "kv_cache": kv_cache,
            }

    rows = []
    for raw in raw_rows:
        batch_size = int(float(raw["max_batch_size"]))
        arrival_rate = float(raw["arrival_rate"])
        kv_budget = int(float(raw["kv_budget"]))
        matches = [
            payload
            for (batch, _gpu_util, rate), payload in metadata_by_key.items()
            if batch == batch_size and abs(rate - arrival_rate) < 1e-9
            and int(
                (
                    payload["kv_cache"].get("gpu_kv_cache_tokens")
                    or (
                        (payload["kv_cache"].get("gpu_kv_cache_blocks") or 0)
                        * (payload["kv_cache"].get("kv_block_size") or 0)
                    )
                )
            )
            == kv_budget
        ]
        if len(matches) != 1:
            raise ValueError(
                "Could not uniquely match stress row to metadata for "
                f"B={batch_size}, arrival_rate={arrival_rate}, kv_budget={kv_budget}."
            )
        payload = matches[0]
        config = payload["config"]
        bench = payload["bench"]
        kv_cache = payload["kv_cache"]
        completed = int(bench.get("completed", 0) or 0)
        failed = int(bench.get("failed", 0) or 0)

        rows.append(
            {
                "arrival_rate": arrival_rate,
                "max_batch_size": batch_size,
                "gpu_memory_utilization": config.get("gpu_memory_utilization"),
                "max_model_len": config.get("max_model_len"),
                "kv_budget": kv_budget,
                "realized_kv_cache_blocks": kv_cache.get("gpu_kv_cache_blocks"),
                "kv_block_size": kv_cache.get("kv_block_size"),
                "num_prompts": config.get("num_prompts"),
                "num_warmups": config.get("num_warmups"),
                "mean_ttft": raw.get("mean_ttft_mean"),
                "p50_ttft": raw.get("p50_ttft_mean"),
                "p95_ttft": raw.get("p95_ttft_mean"),
                "p99_ttft": raw.get("p99_ttft_mean"),
                "mean_tpot": bench.get("mean_tpot_s"),
                "p99_tpot": bench.get("p99_tpot_s"),
                "goodput": raw.get("goodput_mean"),
                "request_throughput": bench.get("request_throughput"),
                "request_goodput_slo_filtered": bench.get(
                    "request_goodput_slo_filtered"
                ),
                "blocking_probability": raw.get("blocking_probability_mean"),
                "preemption_count": config.get("preemption_count"),
                "request_failures": failed,
                "completed": completed,
                # vLLM bench JSON does not always expose these. Keep columns
                # explicit so real runs can fill them when available.
                "max_concurrency": bench.get("max_concurrency", ""),
                "max_active_sequences": bench.get("max_active_sequences", ""),
                "output_throughput": bench.get("output_throughput"),
                "total_token_throughput": bench.get("total_token_throughput"),
                "vllm_version": metadata.get("vllm_version", base.VLLM_VERSION),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STRESS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_local(
    trace: str = "data/sharegpt_trace.parquet",
    output: str = "results/vllm_stress_summary.csv",
    metadata_output: str = "results/vllm_stress_metadata.json",
    raw_output: str = "results/vllm_stress_raw_summary.csv",
    arrival_rates: str = DEFAULT_STRESS_ARRIVAL_RATES,
    batch_sizes: str = DEFAULT_STRESS_BATCH_SIZES,
    gpu_memory_utils: str = DEFAULT_STRESS_GPU_MEMORY_UTILS,
    max_model_len: int = DEFAULT_STRESS_MAX_MODEL_LEN,
    num_prompts: int = DEFAULT_STRESS_NUM_PROMPTS,
    num_warmups: int = DEFAULT_STRESS_NUM_WARMUPS,
    goodput_slo: str = base.DEFAULT_GOODPUT_SLO,
    dry_run: bool = False,
) -> None:
    base.run_local(
        trace=trace,
        output=raw_output,
        metadata_output=metadata_output,
        arrival_rates=arrival_rates,
        batch_sizes=batch_sizes,
        gpu_memory_utils=gpu_memory_utils,
        max_model_len=max_model_len,
        num_prompts=num_prompts,
        num_warmups=num_warmups,
        goodput_slo=goodput_slo,
        smoke=False,
        dry_run=dry_run,
    )
    if dry_run:
        print("stress output after real run:", output)
        print(
            "stress note: the tiny stress grid uses two higher request rates, "
            "B=64, gpu_memory_utilization=0.3, 200 measured requests, and "
            "max_model_len=4096 to probe queueing/KV pressure beyond the "
            "original 64-row sweep."
        )
        return

    _write_stress_summary(
        raw_summary_path=Path(raw_output),
        metadata_path=Path(metadata_output),
        output_path=Path(output),
    )
    print(f"wrote {output}")


@app.local_entrypoint(name="vllm_stress_sweep")
def main(
    trace: str = "data/sharegpt_trace.parquet",
    output: str = "results/vllm_stress_summary.csv",
    metadata_output: str = "results/vllm_stress_metadata.json",
    raw_output: str = "results/vllm_stress_raw_summary.csv",
    arrival_rates: str = DEFAULT_STRESS_ARRIVAL_RATES,
    batch_sizes: str = DEFAULT_STRESS_BATCH_SIZES,
    gpu_memory_utils: str = DEFAULT_STRESS_GPU_MEMORY_UTILS,
    max_model_len: int = DEFAULT_STRESS_MAX_MODEL_LEN,
    num_prompts: int = DEFAULT_STRESS_NUM_PROMPTS,
    num_warmups: int = DEFAULT_STRESS_NUM_WARMUPS,
    goodput_slo: str = base.DEFAULT_GOODPUT_SLO,
    dry_run: bool = False,
) -> None:
    run_local(
        trace=trace,
        output=output,
        metadata_output=metadata_output,
        raw_output=raw_output,
        arrival_rates=arrival_rates,
        batch_sizes=batch_sizes,
        gpu_memory_utils=gpu_memory_utils,
        max_model_len=max_model_len,
        num_prompts=num_prompts,
        num_warmups=num_warmups,
        goodput_slo=goodput_slo,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run a vLLM stress sweep for queueing/KV pressure."
    )
    parser.add_argument("--trace", default="data/sharegpt_trace.parquet")
    parser.add_argument("--output", default="results/vllm_stress_summary.csv")
    parser.add_argument(
        "--metadata-output", default="results/vllm_stress_metadata.json"
    )
    parser.add_argument("--raw-output", default="results/vllm_stress_raw_summary.csv")
    parser.add_argument("--arrival-rates", default=DEFAULT_STRESS_ARRIVAL_RATES)
    parser.add_argument("--batch-sizes", default=DEFAULT_STRESS_BATCH_SIZES)
    parser.add_argument("--gpu-memory-utils", default=DEFAULT_STRESS_GPU_MEMORY_UTILS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_STRESS_MAX_MODEL_LEN)
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_STRESS_NUM_PROMPTS)
    parser.add_argument("--num-warmups", type=int, default=DEFAULT_STRESS_NUM_WARMUPS)
    parser.add_argument("--goodput-slo", default=base.DEFAULT_GOODPUT_SLO)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_local(
        trace=args.trace,
        output=args.output,
        metadata_output=args.metadata_output,
        raw_output=args.raw_output,
        arrival_rates=args.arrival_rates,
        batch_sizes=args.batch_sizes,
        gpu_memory_utils=args.gpu_memory_utils,
        max_model_len=args.max_model_len,
        num_prompts=args.num_prompts,
        num_warmups=args.num_warmups,
        goodput_slo=args.goodput_slo,
        dry_run=args.dry_run,
    )
