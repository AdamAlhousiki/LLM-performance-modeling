# Interview Cheat Sheet: LLM Serving Performance

## Memorize These Three Points

1. **Goal:** Predict how arrival rate, batch size, and KV memory affect LLM serving performance.
2. **Main result:** Doubling load from 12 to 24 requests/s increased mean TTFT about 17 times. TPOT increased about 7%.
3. **Lesson:** Average behavior is easier to predict than tail latency under overload.

## My 60-Second Answer

"Our project studied performance prediction for continuous-batching LLM servers.
I focused on the empirical side. I built a SimPy simulator and a Modal/vLLM
measurement pipeline. We compared queueing models against simulation, then
backtested them against saved GPU measurements.

The server has a compute limit and a KV-memory limit. Requests can wait for
service even when the server accepts them. We measured throughput and the time
before the first output token.

Our strongest hardware result came from a stress test. Increasing load from
12 to 24 requests/s raised mean first-token delay from 56 ms to 942 ms.
Output-token time increased only from 11.38 to 12.22 ms. This supported a
queueing explanation.

The models predicted ordinary-load averages better than overloaded tails.
The main lesson was that matching throughput does not mean we have accurately
modeled the scheduler or tail latency."

Use "I" for your simulator and measurement work. Use "we" for team contributions.

## Pipeline

```text
Synthetic request lengths -> SimPy -> simulated metrics
Same service assumptions -> analytical models -> predicted metrics
Simulation vs predictions -> error tables and tail calibration

ShareGPT -> fixed replay trace -> Modal A10G -> vLLM/Qwen -> GPU metrics
Saved GPU inputs + separate decode measurement -> retrospective backtest
```

The baseline simulator uses synthetic lengths. Only the new replay adapter
uses the ShareGPT length pairs. Do not claim the original workloads were identical.

## Terms You Must Know

| Term | Meaning |
|---|---|
| Inference | Run the trained model to produce output. |
| Prefill | Process the prompt before output starts. |
| Decode | Generate output tokens over successive iterations. |
| Continuous batching | Refill active positions as requests finish. |
| KV cache | Store attention keys and values from previous tokens. |
| Lambda | Average arrival rate, in requests/s. |
| B | Maximum number of active requests. |
| K | KV capacity, represented in token units. |
| TTFT | Arrival-to-first-token time. Includes waiting and prefill. |
| p99 TTFT | 99% of measured requests have TTFT at or below this value. |
| TPOT | Time per output token, excluding the first token. |
| Goodput here | Completed requests / measurement duration. No strict latency SLO. |
| Blocking in SimPy | Rejected arrivals / total arrivals. |
| Preemption in vLLM | Interrupt active work, for example to recover KV capacity. |

One token is not necessarily one word. Peak outstanding requests include
waiting requests. They are not all active GPU sequences.

## How the Simulator Works

1. Sample an exponential interarrival time. This produces Poisson arrivals.
2. Sample lognormal prompt and output lengths. Some requests are much longer.
3. Set request KV demand to prompt length plus output length.
4. Reserve the full demand at admission. Reject if it does not fit.
5. Put accepted requests in a waiting queue. Fill up to B active positions.
6. Process prefill. Then advance each active request by one token per decode step.
7. Release KV memory on completion. Record request timestamps.

SimPy advances a simulated clock. It does not run the neural network.
The latency function assigns a time cost to service steps.
New prefill work can pause decode in this simplified scheduler.
Baseline TTFT statistics exclude rejected and unfinished requests.

## The Analytical Models

**Shared occupancy model:** Replace request sizes with averages.

```text
N = floor(K / average request KV)    approximate admitted capacity
C = min(B, N)                      maximum active count
n                                  current admitted request count
active = min(n, C)
departure rate = active / estimated request service time
```

An admitted arrival moves n to n+1. A completion moves n to n-1.
The stationary probability pi[n] gives the long-run probability of each count.
Blocking is pi[N]. Accepted rate is lambda * (1 - pi[N]).
Mean waiting time comes from mean queue length / accepted rate.
Mean TTFT adds first-token service time to this waiting time.

**Final closed-form:** Uses occupancy probabilities and an Erlang waiting-time
mixture. An arrival may need several completions before it gets an active slot.
Combine these cases to estimate TTFT percentiles.

**Markov-lite:** Uses equivalent occupancy and service rates for averages.
It uses a simpler exponential tail with a utilization-dependent scale.
Its average predictions match closed-form; its tail predictions differ.

**Calibrated-tail:** Fit separate linear regressions for p95 and p99:

```text
target = log(simulator tail / Markov tail)
corrected tail = Markov tail * exp(predicted correction)
```

Features include lambda, B, K, predicted blocking, and predicted goodput.
Standardization uses training data only. The split has 36 training rows and
12 test rows. Every fourth sorted row is test. With four KV values, this
holds out the smallest KV value. It is not a random split.
Mean TTFT, goodput, and blocking remain unchanged.
This correction is trained on simulation. It was not validated on the GPU grid.

The older M/M/C approximation is legacy. The multi-class solver and policy
modules are additional work, not the source of the main reported comparison.

## Numbers and Their Meaning

| Test | Result | Scope |
|---|---|---|
| Analytical vs SimPy | 8.9% mean goodput error | 48 simulator configurations. |
| Analytical vs GPU | 15.9% mean TTFT error | Original 64 hardware configurations only. |
| Analytical vs GPU | 8.5% mean throughput error | Stationary prediction vs finite benchmark. |
| Markov-lite vs GPU | 40.9% mean p99 error | Original 64 hardware configurations only. |
| Tail calibration vs SimPy | Median p99 error: 52.14% -> 34.18% | Overall, including training rows. |

Relative error = abs(prediction - reference) / reference.
A 3 s prediction against a 1 s reference has 200% error.
Mean error is sensitive to extreme points. Median error describes a typical point.

## Hardware Setup and Stress Result

- Qwen/Qwen2.5-1.5B-Instruct, vLLM 0.22.0, one Modal A10G.
- Original sweep: 64 rows; 300 measured requests and 8 warmups per row.
- Stress: two rows; B=64, GPU utilization=0.3, maximum length=4096.
- Stress: 200 measured requests and 4 warmups per row.
- Stress realized KV: 108,704 tokens, or 6,794 blocks of 16 tokens.

| Offered requests/s | Mean TTFT | p99 TTFT | TPOT | Completed requests/s |
|---:|---:|---:|---:|---:|
| 12 | 0.056 s | 0.320 s | 11.38 ms/token | 9.10 |
| 24 | 0.942 s | 2.485 s | 12.22 ms/token | 11.70 |

Both stress rows recorded zero failures and zero preemptions.
Say "evidence of queueing pressure." Do not say "proved KV exhaustion."
TTFT also includes prefill and software overhead.

## The GPU Backtest: What Was Matched

Matched arrival rate, B, realized K, and request-length inputs.
Used an independent decode microbenchmark. Did not fit serving TTFT results.
GPU token counts implied 29 extra prompt tokens per request; these were included.
The replay used three arrival seeds and drained all requests.

**What still differs:** Prefill remains approximate. Exact arrival timestamps
were not saved. The decode curve covers batches 1..32. Larger batches require
extrapolation. vLLM uses different scheduling, prefix caching, and KV allocation.
Analytical accepted rate is stationary; GPU throughput includes a finite drain.

**Overload failure:** At 24 requests/s, analytical mean TTFT was 11.46 s.
Actual GPU mean TTFT was 0.942 s. Do not apply the 16% result to this row.

## Likely Interview Questions

**Why use a GPU at all?** To measure actual timing and check whether real
serving shows the queueing behavior suggested by the models.

**Why not only benchmark vLLM?** Simulation lets us isolate variables cheaply.
Analytical models make the assumptions and predicted relationships explicit.

**Why does more KV sometimes increase TTFT?** More capacity admits more work.
Blocking can decrease while waiting time increases.

**Why are tails harder?** They depend on rare congested states and long requests.
An average request size loses that information.

**Does this make inference faster?** It supports performance prediction and
capacity planning. It does not introduce a faster inference kernel.

**What was technically difficult?** Comparable workloads, warmup treatment,
realized KV measurement, and consistent throughput and failure definitions.

**What would you improve first?** Measure prefill independently. Save exact
arrival timestamps. Replay the same workload before adding model complexity.

**Why trust your conclusions?** We checked controlled simulation and actual GPU
measurements. We also reported model failures and remaining mismatches.

## Claims to Keep Precise

- The 16% result is mean relative error for mean TTFT, not p99 accuracy.
- The GPU result excludes stress rows. The overloaded model predictions were poor.
- The original hardware grid has one run per configuration, not repeated GPU seeds.
- Stable/overloaded labels use simulator blocking below/above 5%; they are practical labels.
- Low blocking does not prove mathematical stability or absence of queueing.
- Goodput here counts completions. It is not strict SLO-filtered goodput.
- Calibration improved overall medians. Held-out overload errors remained high.
- Do not claim an optimal DP admission controller. The policy code uses a simpler decision rule.

## Final Sentence

"The project taught me that throughput agreement is not enough. A useful
serving model must also explain waiting, memory pressure, and latency tails."
