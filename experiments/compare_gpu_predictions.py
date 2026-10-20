"""Retrospective comparison with saved vLLM data; never launches a GPU job."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytical.approximation import GENERATION_SIGMA, PROMPT_SIGMA
from analytical.closed_form import two_stage_metrics
from analytical.fit_service_rate import load_latency_curve
from analytical.markov_chain import markov_lite_metrics
from simulator.metrics import summarize
from simulator.simpy_simulator import ActiveRequest, ContinuousBatchingSimulator, RequestRecord


METRICS = ["mean_ttft", "p95_ttft", "p99_ttft", "goodput"]
ROOT = Path(__file__).resolve().parents[1]


class TraceReplay(ContinuousBatchingSimulator):
    """Reuse baseline scheduling; replace only the arrival source for this run."""

    def __init__(self, lengths, prompt_overhead, **kwargs):
        super().__init__(duration=math.inf, **kwargs)
        self.lengths = lengths
        self.prompt_overhead = prompt_overhead

    def _arrival_process(self):
        for request_id, (prompt, generation) in enumerate(self.lengths):
            if request_id:
                yield self.env.timeout(self.rng.exponential(1.0 / self.arrival_rate))
            prompt = int(prompt) + self.prompt_overhead
            generation = int(generation)
            request = RequestRecord(
                request_id, self.env.now, prompt, generation, prompt + generation, False
            )
            if self.reserved_kv + request.kv_required <= self.kv_budget:
                request.accepted = True
                request.admission_time = self.env.now
                self.reserved_kv += request.kv_required
                self.waiting.append(ActiveRequest(record=request))
                self._wake_server()
            else:
                request.rejected_time = self.env.now
            self.records.append(request)

    def run_to_completion(self, latency_curve):
        # The measured curve applies only inside this isolated replay. The
        # baseline module is restored automatically, including after an error.
        with patch("simulator.simpy_simulator.iteration_latency", latency_curve.iteration_latency):
            self.env.process(self._arrival_process())
            self.env.process(self._server_process())
            self.env.run()
        metrics = summarize(self.records, self.env.now)
        metrics["goodput"] = metrics["goodput_rps"]
        metrics["duration_s"] = self.env.now
        return metrics


def saved_configs(results_dir):
    for source, stem in [("original", "vllm_sweep"), ("stress", "vllm_stress")]:
        metadata = json.loads((results_dir / f"{stem}_metadata.json").read_text())
        summary = pd.read_csv(results_dir / f"{stem}_summary.csv")
        for config in metadata["configs"]:
            kv = config["kv_cache"]["gpu_kv_cache_tokens"]
            for bench in config["bench_runs"]:
                matches = summary[
                    (summary["arrival_rate"] == bench["arrival_rate"])
                    & (summary["max_batch_size"] == config["max_num_seqs"])
                    & (summary["kv_budget"] == kv)
                ]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous saved result: {source}, {config}, {bench}")
                row = matches.iloc[0]
                measured = {
                    metric: float(row[f"{metric}_mean" if source == "original" else metric])
                    for metric in METRICS
                }
                yield source, config, bench, measured


def compare(args):
    trace = json.loads((ROOT / "data/length_dist.json").read_text())
    curve_path = ROOT / "results/latency_curve.json"
    curve = load_latency_curve(curve_path)
    calibration = json.loads(curve_path.read_text())
    observed_batches = [row["batch_size"] for row in calibration["measurements"]]
    rows = []
    for source, config, bench, measured in saved_configs(ROOT / "results"):
        count = config["num_prompts"]
        lengths = list(zip(trace["prompt_len"][:count], trace["gen_len"][:count]))
        if len(lengths) != count:
            raise ValueError("Canonical trace is too short.")
        throughput = bench["request_throughput"]
        gpu_gen_mean = bench["output_throughput"] / throughput
        gpu_prompt_mean = (bench["total_token_throughput"] - bench["output_throughput"]) / throughput
        if not math.isclose(np.mean([g for _, g in lengths]), gpu_gen_mean, abs_tol=1e-5):
            raise ValueError("Trace output lengths do not match saved GPU token counts.")
        overhead_float = gpu_prompt_mean - np.mean([p for p, _ in lengths])
        overhead = round(overhead_float)
        if overhead < 0 or abs(overhead_float - overhead) > 1e-5:
            raise ValueError("GPU prompt counts cannot be matched by a constant token overhead.")
        common = dict(
            source=source,
            arrival_rate=bench["arrival_rate"],
            max_batch_size=config["max_num_seqs"],
            kv_budget=config["kv_cache"]["gpu_kv_cache_tokens"],
            num_prompts=count,
            num_warmups=config["num_warmups"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            max_model_len=config["max_model_len"],
            average_prompt_len=gpu_prompt_mean,
            average_gen_len=gpu_gen_mean,
            inferred_prompt_token_overhead=overhead,
            measured_curve_max_batch=max(observed_batches),
            gpu_benchmark_duration_s=count / throughput,
        )
        predictions = []
        # Existing APIs take lognormal medians. Convert empirical means to
        # equivalent medians so their internal expectation matches the trace.
        for name, model in [("closed_form", two_stage_metrics), ("markov_lite", markov_lite_metrics)]:
            result = model(
                common["arrival_rate"], common["max_batch_size"], common["kv_budget"],
                prompt_mean=gpu_prompt_mean / math.exp(PROMPT_SIGMA**2 / 2),
                generation_mean=gpu_gen_mean / math.exp(GENERATION_SIGMA**2 / 2),
                latency_model=curve,
            )
            predictions.append((name, "stationary_accepted_rate", result))
        replay_results = []
        for seed in args.seeds:
            sim = TraceReplay(
                lengths, overhead, arrival_rate=common["arrival_rate"],
                max_batch_size=common["max_batch_size"], kv_budget=common["kv_budget"], seed=seed,
            )
            replay_results.append(sim.run_to_completion(curve))
        replay_mean = {
            metric: float(np.mean([row[metric] for row in replay_results]))
            for metric in METRICS + ["blocking_probability", "duration_s"]
        }
        for metric in METRICS:
            replay_mean[f"{metric}_seed_std"] = float(np.std([row[metric] for row in replay_results]))
        predictions.append(("trace_replay", "completed_requests_per_drained_run_second", replay_mean))
        for name, window, prediction in predictions:
            output = dict(common, method=name, goodput_window=window)
            for metric in METRICS:
                predicted, observed = prediction[metric], measured[metric]
                if not math.isfinite(predicted):
                    raise ValueError(f"Nonfinite prediction: {name}, {common}, {metric}")
                output[f"{metric}_prediction"] = predicted
                output[f"{metric}_gpu"] = observed
                output[f"{metric}_relative_error"] = abs(predicted - observed) / observed if observed else math.nan
                output[f"{metric}_seed_std"] = prediction.get(f"{metric}_seed_std", math.nan)
            output["predicted_blocking_probability"] = prediction["blocking_probability"]
            output["gpu_request_failures"] = bench["failed"]
            output["replay_duration_s"] = prediction.get("duration_s", math.nan)
            rows.append(output)
    return pd.DataFrame(rows)


def save_plot(predictions, output):
    stress = predictions[predictions["source"] == "stress"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, metric, label in zip(axes, ["mean_ttft", "p99_ttft", "goodput"], ["Mean TTFT (s)", "p99 TTFT (s)", "Completed requests/s"]):
        hardware = stress.drop_duplicates("arrival_rate").sort_values("arrival_rate")
        axis.plot(hardware["arrival_rate"], hardware[f"{metric}_gpu"], "ko-", label="Measured GPU")
        for method, group in stress.groupby("method"):
            group = group.sort_values("arrival_rate")
            axis.plot(group["arrival_rate"], group[f"{metric}_prediction"], "o--", label=method.replace("_", " "))
        axis.set(xlabel="Offered requests/s", ylabel=label)
        if metric != "goodput":
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2", type=lambda value: [int(item) for item in value.split(",")])
    args = parser.parse_args()
    predictions = compare(args)
    results = ROOT / "results"
    predictions.to_csv(results / "gpu_prediction_comparison.csv", index=False)
    errors = []
    for (source, method), group in predictions.groupby(["source", "method"]):
        for metric in METRICS:
            values = group[f"{metric}_relative_error"]
            errors.append(dict(source=source, method=method, metric=metric, n_configs=len(group), mean_relative_error=values.mean(), median_relative_error=values.median()))
    error_table = pd.DataFrame(errors)
    error_table.to_csv(results / "gpu_prediction_errors.csv", index=False)
    save_plot(predictions, ROOT / "figures/gpu_prediction_stress_comparison.png")
    print("Saved results/gpu_prediction_comparison.csv and results/gpu_prediction_errors.csv")
    print("Saved figures/gpu_prediction_stress_comparison.png")
    print("66 saved hardware configurations; no GPU calls; no fit to serving TTFT results.")
    print("Prompt token overhead:", sorted(predictions["inferred_prompt_token_overhead"].unique()))
    print(error_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nStress replay vs GPU:")
    selected = predictions[(predictions["source"] == "stress") & (predictions["method"] == "trace_replay")]
    print(selected[["arrival_rate"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ["prediction", "gpu"]]].to_string(index=False))
    print("\nLimits: arrival timestamps/seed not saved; only request lengths are replayed.")
    print("Prefill remains approximate; the measured decode curve covers batches 1..32.")
    print("GPU prefix caching/V1 scheduling differ; analytical goodput is a stationary rate.")


if __name__ == "__main__":
    main()
