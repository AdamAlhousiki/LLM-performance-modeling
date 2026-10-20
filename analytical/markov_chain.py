from __future__ import annotations

import math

from analytical.approximation import (
    GENERATION_SIGMA,
    MAX_UNSTABLE_WAIT_SECONDS,
    PROMPT_SIGMA,
    lognormal_expected_value,
)
from analytical.fit_service_rate import LatencyModel, get_iteration_latency


def _service_time(
    active_batch: int,
    avg_prompt: float,
    avg_generation: float,
    latency_model: LatencyModel | None = None,
) -> float:
    if active_batch <= 0:
        return 0.0
    prefill = get_iteration_latency(
        latency_model, active_batch, int(round(avg_prompt)), "prefill"
    )
    decode = get_iteration_latency(latency_model, active_batch, 1, "decode")
    return prefill + avg_generation * decode


def _first_token_time(
    active_batch: int,
    avg_prompt: float,
    latency_model: LatencyModel | None = None,
) -> float:
    if active_batch <= 0:
        active_batch = 1
    return get_iteration_latency(
        latency_model, active_batch, int(round(avg_prompt)), "prefill"
    ) + get_iteration_latency(latency_model, active_batch, 1, "decode")


def birth_death_stationary(
    arrival_rate: float,
    death_rates: list[float],
) -> list[float]:
    """Return pi[0..N] for a finite birth-death chain.

    Births are lambda for all states below N and 0 at N. death_rates[i] is
    mu_{i+1}. Ratios are normalized in log-space so overload cases remain
    finite instead of overflowing.
    """
    log_weights = [0.0]
    for mu_n in death_rates:
        if arrival_rate <= 0:
            log_weights.append(-math.inf)
        elif mu_n <= 0:
            log_weights.append(math.inf)
        else:
            log_weights.append(log_weights[-1] + math.log(arrival_rate / mu_n))

    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        pi = [0.0] * len(log_weights)
        pi[-1] = 1.0
        return pi

    offset = max(finite)
    weights = [
        0.0 if value == -math.inf else math.exp(min(value - offset, 700.0))
        for value in log_weights
    ]
    total = sum(weights)
    if total <= 0:
        return [1.0] + [0.0] * (len(log_weights) - 1)
    return [weight / total for weight in weights]


def markov_lite_metrics(
    arrival_rate: float,
    max_batch_size: int,
    kv_budget: int,
    *,
    prompt_mean: float = 256.0,
    generation_mean: float = 128.0,
    latency_model: LatencyModel | None = None,
) -> dict[str, float]:
    """One-dimensional Markov-chain-lite approximation.

    State n is the number of admitted requests in the system. The finite state
    cap N is implied by average KV demand. This is intentionally simpler than
    the multi-class Markov solver: it keeps the exact birth-death stationary
    distribution over occupancy, then estimates latency via Little's law and
    an exponential tail approximation.
    """
    avg_prompt = lognormal_expected_value(prompt_mean, PROMPT_SIGMA)
    avg_generation = lognormal_expected_value(generation_mean, GENERATION_SIGMA)
    avg_kv_per_request = avg_prompt + avg_generation

    n_max = max(1, int(kv_budget // avg_kv_per_request))
    active_cap = max(1, min(max_batch_size, n_max))

    death_rates = []
    for n in range(1, n_max + 1):
        active_batch = min(n, active_cap)
        service = _service_time(active_batch, avg_prompt, avg_generation, latency_model)
        death_rates.append(active_batch / service if service > 0 else 0.0)

    pi = birth_death_stationary(arrival_rate, death_rates)
    blocking_probability = pi[n_max]
    accepted_rate = arrival_rate * (1.0 - blocking_probability)
    goodput = accepted_rate

    mean_number = sum(n * p_n for n, p_n in enumerate(pi))
    mean_latency = (
        min(mean_number / accepted_rate, MAX_UNSTABLE_WAIT_SECONDS)
        if accepted_rate > 0
        else MAX_UNSTABLE_WAIT_SECONDS
    )

    busy_mass = 1.0 - pi[0]
    if busy_mass > 1e-12:
        mean_active_when_busy = (
            sum(min(n, active_cap) * pi[n] for n in range(1, n_max + 1)) / busy_mass
        )
        typical_active = max(1, min(active_cap, int(round(mean_active_when_busy))))
    else:
        typical_active = 1

    first_token_service = _first_token_time(
        typical_active, avg_prompt, latency_model
    )
    mean_queue_occupancy = sum(max(n - active_cap, 0) * pi[n] for n in range(n_max + 1))
    mean_queue_wait = (
        min(mean_queue_occupancy / accepted_rate, MAX_UNSTABLE_WAIT_SECONDS)
        if accepted_rate > 0
        else MAX_UNSTABLE_WAIT_SECONDS
    )
    mean_ttft = min(first_token_service + mean_queue_wait, MAX_UNSTABLE_WAIT_SECONDS)

    full_service = _service_time(active_cap, avg_prompt, avg_generation, latency_model)
    full_capacity = active_cap / full_service if full_service > 0 else 0.0
    rho = min(arrival_rate / full_capacity, 0.99) if full_capacity > 0 else 0.99
    tail_scale = max(mean_queue_wait, first_token_service * 0.25, 1e-9)
    tail_scale *= 1.0 + rho / max(1.0 - rho, 1e-6)

    def ttft_quantile(q: float) -> float:
        return min(
            first_token_service - math.log(max(1.0 - q, 1e-12)) * tail_scale,
            MAX_UNSTABLE_WAIT_SECONDS,
        )

    return {
        "arrival_rate": arrival_rate,
        "max_batch_size": max_batch_size,
        "kv_budget": kv_budget,
        "mean_ttft": mean_ttft,
        "p50_ttft": ttft_quantile(0.50),
        "p95_ttft": ttft_quantile(0.95),
        "p99_ttft": ttft_quantile(0.99),
        "goodput": goodput,
        "blocking_probability": blocking_probability,
        "mean_latency": mean_latency,
        "mean_number_in_system": mean_number,
        "accepted_rate": accepted_rate,
        "n_states": n_max + 1,
    }
