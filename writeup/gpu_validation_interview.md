# GPU Comparison and Interview Notes

This is an additional retrospective analysis of the saved hardware results.
It does not replace the final report or the original simulator comparisons.
No new GPU run was launched. The serving TTFT results were not used to fit
the latency curve or the analytical models.

## What Was Matched

- All 64 original hardware configurations and both stress configurations.
- Offered arrival rate, maximum active sequences, and realized KV tokens.
- The corresponding first 300 or 200 requests from the canonical trace.
- Recorded generation lengths, checked against hardware output-token counts.
- Prompt lengths plus a constant 29-token overhead. Hardware token counts
  imply this overhead; the comparison does not assume its exact origin.
- The independently measured decode curve in `results/latency_curve.json`.

The analytical models use the empirical mean prompt and output lengths.
An isolated simulator adapter replays the individual length pairs. It retains
the baseline scheduling logic, uses three arrival seeds (0, 1, 2), and drains
all requests before calculating completed requests per second. It applies the
measured decode curve only during these additional runs. Original simulator
code and saved results remain unchanged.

## Quantitative Results

These are mean absolute relative errors against the 64 original GPU rows.
Each configuration has equal weight. Lower is better.

| Metric | Closed-form | Markov-lite |
|---|---:|---:|
| Mean TTFT | 15.9% | 15.9% |
| p95 TTFT | 47.5% | 34.0% |
| p99 TTFT | 59.1% | 40.9% |
| Throughput | 8.5% | 8.5% |

The analytical throughput prediction is a stationary accepted rate. The GPU
throughput is completed requests divided by a finite benchmark duration,
including the drain after arrivals stop. That window difference contributes
to throughput error; these percentages are a first backtest, not an exact
equivalence of measurement procedures.

The finite trace replay uses a closer throughput window. Its mean throughput
error is 0.36% on the original grid. However, mean TTFT error is 47.0% and
p99 error is 276.2%. Agreement in throughput does not imply scheduler fidelity.
At these low offered loads, throughput is mainly controlled by the arrival
rate, which also makes the throughput result easier to match.

The earlier 8.9% goodput error is analytical vs SimPy. The new 8.5% throughput
error is analytical vs the original GPU sweep. They refer to different tests.

## Stress Results and Model Failure

| Offered rate | GPU mean TTFT | Analytical mean TTFT | Trace-replay mean TTFT |
|---:|---:|---:|---:|
| 12 requests/s | 0.0558 s | 0.0502 s | 1.1411 s |
| 24 requests/s | 0.9424 s | 11.4568 s | 4.0582 s |

The existing approximations do not accurately predict the overloaded row.
The hardware observation remains useful: mean TTFT increased about 17 times,
while mean TPOT increased about 7%. This supports queueing before first token
as the main explanation, but TTFT also contains prefill and other overhead.
Both stress rows completed 200 requests. They recorded zero failures and
zero preemptions. This is evidence of queueing pressure, not KV exhaustion.

## Remaining Differences

1. Prefill timing remains the simulator approximation. The microbenchmark
   measured streamed decode intervals, not prefill or exact GPU kernel time.
2. The decode curve was measured at batch sizes 1 through 32. Larger active
   batches require extrapolation.
3. Request arrival timestamps and benchmark arrival seeds were not saved.
   The replay matches length order and Poisson arrival rate, not exact timing.
4. GPU warmups, prefix caching, V1 chunked prefill, and asynchronous scheduling
   differ from the simplified simulator. Measured intertoken intervals also
   include streaming overhead.
5. Simulator admission reserves full request KV at arrival. vLLM has different
   memory allocation and queueing semantics. No blocking/failure error is
   calculated between those two different definitions.
6. The 64 GPU rows have one hardware run each. Replay seed variation is not
   a confidence interval for repeated GPU experiments.
7. Calibrated-tail was trained on simulator configurations. It was not applied
   to this different GPU grid.

## A Short Interview Answer

"I built the simulator and GPU measurement pipeline for an LLM serving
performance project. We compared queueing models with simulation, then
backtested existing predictions against saved vLLM measurements. We aligned
the request lengths, batch limits, and realized KV capacity, and used a
separately measured decode curve. Across the original 64 GPU configurations,
mean TTFT error was about 16%. Tail errors were larger. Under high load,
the model became much less accurate. The hardware stress test showed why this
matters: first-token delay increased about 17 times, while per-token time rose
only about 7%. The main lesson was that matching throughput is easier than
matching queueing and tail latency."

## Answers to Likely Follow-Ups

**What did the GPU experiment add?** It supplied actual service timings and
tested whether the queueing trends occurred in a real serving engine.

**Did you get 16% accuracy everywhere?** No. That is mean relative error for
mean TTFT over the original 64 configurations. It excludes the stress rows.

**Did you fit the model to those TTFT measurements?** No. Decode timing came
from a separate microbenchmark. Workload lengths came from the trace and token
counts. Prefill was not fitted, and its approximation remains a limitation.

**Why do the two models share average predictions?** Their occupancy chains
and service rates are equivalent in this comparison. Their tail formulas differ.

**Why was the overload prediction poor?** The simplified prefill scheduler,
average-size assumptions, extrapolated decode curve, and stationary vs finite
measurement windows are plausible sources. This backtest does not isolate
their individual contributions.

**What would you improve first?** Measure prefill independently and preserve
per-request arrival times. Then replay the exact workload before adding model
complexity.

## Artifacts and Reproduction

```powershell
python -m experiments.compare_gpu_predictions
```

- `results/gpu_prediction_comparison.csv`: predictions and references for
  each hardware configuration and method.
- `results/gpu_prediction_errors.csv`: error summaries by original/stress grid.
- `figures/gpu_prediction_stress_comparison.png`: stress predictions vs GPU.

This command runs locally. It does not import Modal or launch GPU functions.
