#!/usr/bin/env bash
set -euo pipefail

cat > /app/advnet_selection.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class TraceCandidate:
    trace_id: str
    true_score: float
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or self.trace_id == "":
            raise ValueError("trace_id must be a non-empty string")

        if not isinstance(self.true_score, (int, float)) or not math.isfinite(self.true_score):
            raise ValueError("true_score must be a finite number")

        if not isinstance(self.noise_std, (int, float)) or not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be a finite non-negative number")


def evaluate_trace(candidate: TraceCandidate, rng: random.Random) -> float:
    return candidate.true_score + rng.gauss(0.0, candidate.noise_std)


def simple_max(
    candidates: list[TraceCandidate],
    rng: random.Random,
) -> TraceCandidate:
    if not candidates:
        raise ValueError("candidates must not be empty")

    best_candidate: TraceCandidate | None = None
    best_score: float | None = None

    for candidate in candidates:
        observed_score = evaluate_trace(candidate, rng)

        if best_candidate is None:
            best_candidate = candidate
            best_score = observed_score
            continue

        assert best_score is not None

        if observed_score > best_score:
            best_candidate = candidate
            best_score = observed_score
        elif observed_score == best_score and candidate.trace_id < best_candidate.trace_id:
            best_candidate = candidate
            best_score = observed_score

    assert best_candidate is not None
    return best_candidate


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rank_candidates(
    candidates: list[TraceCandidate],
    samples: dict[str, list[float]],
) -> list[TraceCandidate]:
    def key(candidate: TraceCandidate) -> tuple[int, float, str]:
        candidate_samples = samples.get(candidate.trace_id, [])

        if not candidate_samples:
            return (1, 0.0, candidate.trace_id)

        return (0, -_mean(candidate_samples), candidate.trace_id)

    return sorted(candidates, key=key)


def multi_round_elimination(
    candidates: list[TraceCandidate],
    budget: int,
    rng: random.Random,
) -> TraceCandidate:
    if not candidates:
        raise ValueError("candidates must not be empty")

    if budget <= 0:
        raise ValueError("budget must be positive")

    survivors = sorted(list(candidates), key=lambda candidate: candidate.trace_id)
    samples: dict[str, list[float]] = {
        candidate.trace_id: []
        for candidate in survivors
    }

    evaluations_used = 0

    while evaluations_used < budget:
        if len(survivors) > 5:
            full_round_completed = True

            for candidate in sorted(survivors, key=lambda candidate: candidate.trace_id):
                if evaluations_used >= budget:
                    full_round_completed = False
                    break

                samples[candidate.trace_id].append(evaluate_trace(candidate, rng))
                evaluations_used += 1

            if not full_round_completed:
                break

            ranked = _rank_candidates(survivors, samples)
            keep_count = math.ceil(len(survivors) / 2)
            survivors = sorted(
                ranked[:keep_count],
                key=lambda candidate: candidate.trace_id,
            )
        else:
            for candidate in sorted(survivors, key=lambda candidate: candidate.trace_id):
                if evaluations_used >= budget:
                    break

                samples[candidate.trace_id].append(evaluate_trace(candidate, rng))
                evaluations_used += 1

    ranked_finalists = _rank_candidates(survivors, samples)
    return ranked_finalists[0]


def select_best_trace(
    candidates: list[TraceCandidate],
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