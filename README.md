# Queueing-Theoretic Performance Modeling for Continuous-Batching LLM Inference

EE384S final project repository for modeling and measuring continuous-batching
LLM inference under finite KV-cache capacity.

The project combines three tracks:

- a SimPy simulator for queueing-level continuous batching,
- analytical approximations for TTFT, goodput, and blocking,
- Modal/vLLM measurements on `Qwen/Qwen2.5-1.5B-Instruct`.

The main question is how arrival rate, batch width, request length, and KV-cache
capacity affect time to first token (TTFT), tail latency, goodput, and blocking
or preemption behavior.

## Repository Structure

| path | purpose |
|---|---|
| `simulator/` | baseline SimPy continuous-batching simulator and metrics |
| `analytical/` | closed-form, Markov, and measured-service-rate analytical models |
| `experiments/` | sweeps, plotting, ShareGPT preprocessing, Modal/vLLM harnesses |
| `infra/` | small Modal/vLLM smoke app |
| `data/` | canonical replay traces and empirical length distributions |
| `results/` | CSV/JSON outputs from simulator, analytical, and vLLM runs |
| `figures/` | generated report figures |
| `writeup/` | model notes and report-ready result summaries |

## Key Results So Far

### Analytical vs SimPy

The final analytical comparison uses the 48-row SimPy grid shared by
`results/sweep_summary.csv`, `results/closed_form_summary.csv`, and
`results/markov_summary.csv`. `results/comparison_summary.csv` is a legacy
first-approximation comparison and is not the source for the final report
tables.

| metric | closed-form mean error | Markov-lite mean error | closed-form median error | Markov-lite median error |
|---|---:|---:|---:|---:|
| mean TTFT | 0.232 | 0.232 | 0.095 | 0.095 |
| p95 TTFT | 0.457 | 2.095 | 0.449 | 0.448 |
| p99 TTFT | 0.632 | 2.052 | 0.595 | 0.521 |
| goodput | 0.089 | 0.089 | 0.087 | 0.087 |
| blocking probability | 0.490 | 0.490 | 0.004 | 0.004 |

The calibrated-tail model only modifies p95/p99 TTFT. It improves median p95
relative error from 0.448 to 0.233 and median p99 relative error from 0.521 to
0.342, while leaving mean TTFT, goodput, and blocking unchanged.

### Modal/vLLM Sweep

The vLLM sweep completed 64 rows on Modal/A10G:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM version: `0.22.0`
- arrival rates: `0.5,1,2,3,4,5,6,8` req/s
- max sequence counts: `64,128`
- GPU memory utilization: `0.3,0.5,0.7,0.9`
- realized KV cache: `104,336` to `614,736` tokens
- benchmark size: 300 measured requests plus 8 warmup requests per row

Headline hardware measurements:

- max observed goodput: `6.74 req/s`
- worst observed p99 TTFT: `0.185 s`
- mean TTFT stayed below `0.061 s`
- mean TPOT ranged from about `8.3` to `10.2 ms/token`
- request failures and preemption count were zero in this measured grid

The current vLLM sweep validates the full measurement pipeline and provides
real latency scale. The stress sweep additionally demonstrates queueing
pressure, but neither sweep triggered KV-cache preemption or request failures.

A tiny two-row stress sweep at 12 and 24 req/s with `max_num_seqs=64`,
`gpu_memory_utilization=0.3`, and `max_model_len=4096` did create queueing
pressure: p99 TTFT rose from `0.320 s` at 12 req/s to `2.485 s` at 24 req/s.
It still produced zero failures and zero preemptions.

More report-ready language is in `writeup/report_results.md`.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Modal experiments require a Modal token and the intended profile. For this
project we used:

```powershell
$env:MODAL_PROFILE='adamco27'
```

## Baseline SimPy Simulator

Run one baseline simulation:

```powershell
.\.venv\Scripts\python.exe -m simulator.run_baseline --duration 200 --arrival-rate 2 --max-batch-size 8 --kv-budget 8192
```

The command writes per-request results to `results/baseline.csv` and prints
TTFT percentiles, mean TTFT, goodput, and blocking probability.

TTFT percentiles are computed over accepted requests that complete before the
simulation ends. Rejected requests and accepted requests still in service at the
horizon are excluded from TTFT percentile calculations.

## SimPy Sweep and Plots

Run the default simulator sweep:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_sweep
```

The default sweep uses duration `1000`, arrival rates `0.5,1,2,3,4,5,6,8`,
max batch sizes `1,2,4,8,16`, KV budgets `2048,4096,8192,16384`, and seeds
`0,1,2,3,4`. It writes `results/sweep_summary.csv`.

Generate report figures:

```powershell
.\.venv\Scripts\python.exe -m experiments.plot_sweep
```

The KV-budget plots hold max batch size fixed at `B=8`. As KV budget increases,
the simulator may admit more requests; this can reduce blocking while
increasing effective load and tail latency, so p99 TTFT can rise even though
fewer requests are rejected.

## Analytical Models

Run the first analytical approximation:

```powershell
.\.venv\Scripts\python.exe -m analytical.run_analytical
```

Compare analytical predictions against SimPy:

```powershell
.\.venv\Scripts\python.exe -m experiments.compare_simulation_analytical
```

Generate comparison plots:

```powershell
.\.venv\Scripts\python.exe -m experiments.plot_comparison
```

The repository also includes richer closed-form and Markov-chain analytical
models, plus optional measured-service-rate injection from
`results/latency_curve.json`.

Run the Markov-chain-lite birth-death model and compare it with SimPy:

```powershell
.\.venv\Scripts\python.exe -m analytical.run_closed_form
.\.venv\Scripts\python.exe -m analytical.run_markov
.\.venv\Scripts\python.exe -m experiments.compare_simulation_markov
.\.venv\Scripts\python.exe -m experiments.plot_markov_comparison
```

This writes:

- `results/closed_form_summary.csv`
- `results/markov_summary.csv`
- `results/markov_comparison_summary.csv`
- `results/analytical_error_table.csv`
- `figures/markov_comparison_relative_errors.png`
- `figures/markov_comparison_mean_ttft.png`
- `figures/markov_comparison_p99_ttft.png`
- `figures/markov_comparison_goodput.png`
- `figures/markov_comparison_blocking_probability.png`

Run the calibrated Markov-tail correction:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_calibrated_tail
.\.venv\Scripts\python.exe -m experiments.plot_calibrated_tail_comparison
```

This fits a lightweight semi-empirical correction to Markov-lite p95/p99 TTFT
log residuals using a deterministic train/test split. It does not change the
simulator, and it does not recalibrate mean TTFT, goodput, or blocking
probability. It writes:

- `results/calibrated_tail_predictions.csv`
- `results/calibrated_tail_error_table.csv`
- `results/calibrated_tail_train_test_error_table.csv`
- `results/calibrated_tail_error_by_regime.csv`
- `figures/calibrated_tail_p95_scatter.png`
- `figures/calibrated_tail_p99_scatter.png`
- `figures/calibrated_tail_relative_errors.png`
- `figures/calibrated_tail_train_test_errors.png`
- `figures/calibrated_tail_error_by_regime.png`

## ShareGPT Trace

Build the canonical replay trace:

```powershell
.\.venv\Scripts\python.exe -m experiments.preprocess_sharegpt --sample-size 5000 --seed 0 --max-model-len 2048
```

This writes:

- `data/sharegpt_trace.parquet`
- `data/length_dist.json`

The trace is the single source of truth for vLLM replay and future
trace-driven simulation.

## Modal/vLLM Microbenchmark

Measure decode step latency on Modal:

```powershell
$env:MODAL_PROFILE='adamco27'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\.venv\Scripts\modal.exe run experiments/microbench_latency.py --output results/latency_curve.json
```

This boots vLLM, measures fixed active batch sizes, fits a sublinear decode
latency curve, and writes `results/latency_curve.json`.

## Modal/vLLM Serving Sweep

Preview the full grid without launching a GPU job:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_vllm_sweep --dry-run
```

Run a small smoke test:

```powershell
$env:MODAL_PROFILE='adamco27'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\.venv\Scripts\modal.exe run experiments/run_vllm_sweep.py --smoke
```

Run the full vLLM sweep:

```powershell
$env:MODAL_PROFILE='adamco27'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\.venv\Scripts\modal.exe run experiments/run_vllm_sweep.py
```

The harness replays `data/sharegpt_trace.parquet` by converting it to vLLM's
custom JSONL benchmark format, preserving `prompt_text` and forcing per-request
output length from `gen_len`. It writes:

- `results/vllm_sweep_summary.csv`
- `results/vllm_sweep_metadata.json`

`results/vllm_sweep_summary.csv` intentionally keeps the same column schema as
`results/sweep_summary.csv`. vLLM version, flags, TPOT, throughput, preemption
count, and realized KV cache metadata are recorded in the metadata JSON.

## Modal/vLLM Stress Sweep

Preview the heavier stress grid without launching a GPU job:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_vllm_stress_sweep --dry-run
```

Run the stress sweep on Modal:

```powershell
$env:MODAL_PROFILE='adamco27'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\.venv\Scripts\modal.exe run experiments/run_vllm_stress_sweep.py::app.vllm_stress_sweep
```

Plot stress results after the Modal run:

```powershell
.\.venv\Scripts\python.exe -m experiments.plot_vllm_stress
```

The stress sweep keeps the original 64-row sweep intact. The default stress
grid is intentionally tiny: arrival rates `12,24`, `max_num_seqs=64`,
`gpu_memory_utilization=0.3`, `max_model_len=4096`, 200 measured requests, and
4 warmups. It probes whether TTFT/TPOT rises or vLLM begins preempting/failing
under pressure without spending hours on a large sweep. It writes:

- `results/vllm_stress_summary.csv`
- `results/vllm_stress_metadata.json`
- `results/vllm_stress_raw_summary.csv`
- `figures/vllm_stress_ttft.png`
- `figures/vllm_stress_tpot.png`
- `figures/vllm_stress_goodput.png`

## GPU Prediction Backtest

Backtest the existing models against saved GPU measurements locally:

```powershell
python -m experiments.compare_gpu_predictions
```

This uses the recorded hardware grid, canonical trace lengths, and independent
decode microbenchmark. It writes `results/gpu_prediction_comparison.csv`,
`results/gpu_prediction_errors.csv`, and
`figures/gpu_prediction_stress_comparison.png`. No GPU job is launched.
Read `writeup/gpu_validation_interview.md` for the comparison results and limits,
including the approximate prefill model and stationary/finite-window mismatch.

## Project Naming

Project title: **Queueing-Theoretic Performance Modeling for Continuous-Batching
LLM Inference**.
