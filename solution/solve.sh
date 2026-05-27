#!/usr/bin/env bash
set -euo pipefail

cat > /app/advnet_selection.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List


@dataclass(frozen=True)
class TraceCandidate:
    trace_id: str
    true_score: float
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise ValueError("trace_id must be a non-empty string")
        if not math.isfinite(self.true_score):
            raise ValueError("true_score must be finite")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")


def evaluate_trace(candidate: TraceCandidate, rng: random.Random) -> float:
    if candidate.noise_std == 0:
        return candidate.true_score
    return candidate.true_score + rng.gauss(0.0, candidate.noise_std)


def simple_max(candidates: List[TraceCandidate], rng: random.Random) -> TraceCandidate:
    if not candidates:
        raise ValueError("candidates must not be empty")

    best_candidate = None
    best_sample = None

    for candidate in sorted(candidates, key=lambda c: c.trace_id):
        sample = evaluate_trace(candidate, rng)
        if (
            best_candidate is None
            or sample > best_sample
            or (sample == best_sample and candidate.trace_id < best_candidate.trace_id)
        ):
            best_candidate = candidate
            best_sample = sample

    return best_candidate


def _mean(samples: list[float]) -> float:
    return sum(samples) / len(samples)


def _rank_key(candidate: TraceCandidate, samples: dict[str, list[float]]) -> tuple:
    candidate_samples = samples.get(candidate.trace_id, [])
    if not candidate_samples:
        return (0, float("-inf"), candidate.trace_id)
    return (1, _mean(candidate_samples), tuple(-ord(ch) for ch in candidate.trace_id))


def _sort_survivors_desc(
    survivors: list[TraceCandidate],
    samples: dict[str, list[float]],
) -> list[TraceCandidate]:
    return sorted(
        survivors,
        key=lambda c: (
            len(samples.get(c.trace_id, [])) > 0,
            _mean(samples[c.trace_id]) if samples.get(c.trace_id) else float("-inf"),
            "".join(chr(255 - ord(ch)) for ch in c.trace_id),
        ),
        reverse=True,
    )


def multi_round_elimination(
    candidates: List[TraceCandidate],
    budget: int,
    rng: random.Random,
) -> TraceCandidate:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if budget <= 0:
        raise ValueError("budget must be positive")

    survivors = sorted(list(candidates), key=lambda c: c.trace_id)
    samples: dict[str, list[float]] = {c.trace_id: [] for c in survivors}
    evaluations = 0

    while evaluations < budget:
        if len(survivors) > 5:
            for candidate in sorted(survivors, key=lambda c: c.trace_id):
                if evaluations >= budget:
                    break
                samples[candidate.trace_id].append(evaluate_trace(candidate, rng))
                evaluations += 1

            ranked = sorted(
                survivors,
                key=lambda c: (
                    len(samples[c.trace_id]) > 0,
                    _mean(samples[c.trace_id]) if samples[c.trace_id] else float("-inf"),
                    "".join(chr(255 - ord(ch)) for ch in c.trace_id),
                ),
                reverse=True,
            )

            keep_count = (len(survivors) + 1) // 2
            survivors = sorted(ranked[:keep_count], key=lambda c: c.trace_id)
        else:
            for candidate in sorted(survivors, key=lambda c: c.trace_id):
                if evaluations >= budget:
                    break
                samples[candidate.trace_id].append(evaluate_trace(candidate, rng))
                evaluations += 1

    final_ranked = sorted(
        survivors,
        key=lambda c: (
            len(samples[c.trace_id]) > 0,
            _mean(samples[c.trace_id]) if samples[c.trace_id] else float("-inf"),
            "".join(chr(255 - ord(ch)) for ch in c.trace_id),
        ),
        reverse=True,
    )

    return final_ranked[0]


def select_best_trace(
    candidates: List[TraceCandidate],
    budget: int,
    rng: random.Random,
    method: str = "mre",
) -> TraceCandidate:
    if budget <= 0:
        raise ValueError("budget must be positive")

    if method == "mre":
        return multi_round_elimination(candidates, budget, rng)
    if method == "simple_max":
        return simple_max(candidates, rng)

    raise ValueError(f"unknown method: {method}")
PY
