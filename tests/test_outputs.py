import random
import pytest

import advnet_selection as mod
from advnet_selection import (
    TraceCandidate,
    evaluate_trace,
    simple_max,
    multi_round_elimination,
    select_best_trace,
)


def test_trace_candidate_validation():
    TraceCandidate("a", 1.0, 0.0)

    with pytest.raises(ValueError):
        TraceCandidate("", 1.0, 0.0)

    with pytest.raises(ValueError):
        TraceCandidate("a", float("inf"), 0.0)

    with pytest.raises(ValueError):
        TraceCandidate("a", 1.0, float("nan"))

    with pytest.raises(ValueError):
        TraceCandidate("a", 1.0, -0.01)


def test_evaluate_trace_zero_noise_is_exact():
    candidate = TraceCandidate("x", 0.75, 0.0)
    rng = random.Random(123)

    assert evaluate_trace(candidate, rng) == 0.75
    assert evaluate_trace(candidate, rng) == 0.75


def test_evaluate_trace_uses_provided_seeded_rng():
    candidate = TraceCandidate("x", 1.0, 0.5)

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    assert evaluate_trace(candidate, rng1) == evaluate_trace(candidate, rng2)


def test_simple_max_selects_highest_observed_score_without_noise():
    candidates = [
        TraceCandidate("a", 0.1, 0.0),
        TraceCandidate("b", 0.9, 0.0),
        TraceCandidate("c", 0.5, 0.0),
    ]

    best = simple_max(candidates, random.Random(1))

    assert best.trace_id == "b"


def test_simple_max_evaluates_each_candidate_once(monkeypatch):
    candidates = [
        TraceCandidate("a", 0.0, 0.0),
        TraceCandidate("b", 0.0, 0.0),
        TraceCandidate("c", 0.0, 0.0),
    ]
    observed = {"a": 0.3, "b": 0.8, "c": 0.5}
    calls = []

    def fake_evaluate(candidate, rng):
        calls.append(candidate.trace_id)
        return observed[candidate.trace_id]

    monkeypatch.setattr(mod, "evaluate_trace", fake_evaluate)

    best = mod.simple_max(candidates, random.Random(1))

    assert best.trace_id == "b"
    assert sorted(calls) == ["a", "b", "c"]
    assert len(calls) == 3


def test_simple_max_tie_breaks_by_lexicographic_trace_id():
    candidates = [
        TraceCandidate("z", 1.0, 0.0),
        TraceCandidate("a", 1.0, 0.0),
        TraceCandidate("m", 1.0, 0.0),
    ]

    best = simple_max(candidates, random.Random(1))

    assert best.trace_id == "a"


def test_simple_max_empty_raises_value_error():
    with pytest.raises(ValueError):
        simple_max([], random.Random(1))


def test_mre_selects_best_candidate_without_noise():
    candidates = [
        TraceCandidate("a", 0.1, 0.0),
        TraceCandidate("b", 0.3, 0.0),
        TraceCandidate("c", 0.9, 0.0),
        TraceCandidate("d", 0.4, 0.0),
        TraceCandidate("e", 0.2, 0.0),
        TraceCandidate("f", 0.8, 0.0),
    ]

    best = multi_round_elimination(candidates, budget=20, rng=random.Random(10))

    assert best.trace_id == "c"


def test_mre_eliminates_bottom_half_not_top_half():
    candidates = [
        TraceCandidate("a", 0.60, 0.0),
        TraceCandidate("b", 0.50, 0.0),
        TraceCandidate("c", 0.40, 0.0),
        TraceCandidate("d", 0.30, 0.0),
        TraceCandidate("e", 0.20, 0.0),
        TraceCandidate("f", 0.10, 0.0),
    ]

    best = multi_round_elimination(candidates, budget=6, rng=random.Random(1))

    assert best.trace_id == "a"


def test_mre_respects_budget_even_when_budget_is_smaller_than_candidates(monkeypatch):
    candidates = [
        TraceCandidate("c", 0.8, 0.0),
        TraceCandidate("a", 0.1, 0.0),
        TraceCandidate("b", 0.9, 0.0),
    ]
    calls = []

    def fake_evaluate(candidate, rng):
        calls.append(candidate.trace_id)
        return candidate.true_score

    monkeypatch.setattr(mod, "evaluate_trace", fake_evaluate)

    best = mod.multi_round_elimination(candidates, budget=2, rng=random.Random(1))

    assert calls == ["a", "b"]
    assert best.trace_id == "b"


def test_mre_never_exceeds_total_evaluation_budget(monkeypatch):
    candidates = [
        TraceCandidate(f"t{i}", float(i), 0.0)
        for i in range(10)
    ]
    calls = []

    def fake_evaluate(candidate, rng):
        calls.append(candidate.trace_id)
        return candidate.true_score

    monkeypatch.setattr(mod, "evaluate_trace", fake_evaluate)

    mod.multi_round_elimination(candidates, budget=7, rng=random.Random(1))

    assert len(calls) == 7


def test_mre_uses_running_mean_not_latest_sample(monkeypatch):
    candidates = [
        TraceCandidate("a", 0.0, 0.0),
        TraceCandidate("b", 0.0, 0.0),
    ]
    samples = {
        "a": [1.0, 1.0, 1.0],
        "b": [0.0, 0.0, 2.0],
    }

    def fake_evaluate(candidate, rng):
        return samples[candidate.trace_id].pop(0)

    monkeypatch.setattr(mod, "evaluate_trace", fake_evaluate)

    best = mod.multi_round_elimination(candidates, budget=6, rng=random.Random(1))

    assert best.trace_id == "a"


def test_mre_tie_breaks_final_result_by_trace_id():
    candidates = [
        TraceCandidate("b", 1.0, 0.0),
        TraceCandidate("a", 1.0, 0.0),
        TraceCandidate("c", 1.0, 0.0),
    ]

    best = multi_round_elimination(candidates, budget=9, rng=random.Random(1))

    assert best.trace_id == "a"


def test_mre_candidates_with_no_samples_rank_below_sampled_candidates(monkeypatch):
    candidates = [
        TraceCandidate("a", 0.1, 0.0),
        TraceCandidate("b", 0.2, 0.0),
        TraceCandidate("c", 999.0, 0.0),
    ]
    calls = []

    def fake_evaluate(candidate, rng):
        calls.append(candidate.trace_id)
        return candidate.true_score

    monkeypatch.setattr(mod, "evaluate_trace", fake_evaluate)

    best = mod.multi_round_elimination(candidates, budget=1, rng=random.Random(1))

    assert calls == ["a"]
    assert best.trace_id == "a"


def test_mre_is_deterministic_for_same_rng_seed():
    candidates = [
        TraceCandidate("a", 0.4, 0.2),
        TraceCandidate("b", 0.5, 0.2),
        TraceCandidate("c", 0.6, 0.2),
        TraceCandidate("d", 0.7, 0.2),
        TraceCandidate("e", 0.8, 0.2),
        TraceCandidate("f", 0.9, 0.2),
    ]

    best1 = multi_round_elimination(candidates, budget=40, rng=random.Random(99))
    best2 = multi_round_elimination(candidates, budget=40, rng=random.Random(99))

    assert best1.trace_id == best2.trace_id


def test_mre_does_not_mutate_input_list():
    candidates = [
        TraceCandidate("c", 0.3, 0.0),
        TraceCandidate("a", 0.1, 0.0),
        TraceCandidate("b", 0.2, 0.0),
    ]
    original_order = [candidate.trace_id for candidate in candidates]

    multi_round_elimination(candidates, budget=10, rng=random.Random(1))

    assert [candidate.trace_id for candidate in candidates] == original_order


def test_mre_empty_candidates_and_bad_budget_raise_value_error():
    with pytest.raises(ValueError):
        multi_round_elimination([], budget=10, rng=random.Random(1))

    with pytest.raises(ValueError):
        multi_round_elimination([TraceCandidate("a", 1.0)], budget=0, rng=random.Random(1))


def test_select_best_trace_defaults_to_mre():
    candidates = [
        TraceCandidate("a", 0.2, 0.0),
        TraceCandidate("b", 0.9, 0.0),
    ]

    best = select_best_trace(candidates, budget=10, rng=random.Random(1))

    assert best.trace_id == "b"


def test_select_best_trace_supports_simple_max():
    candidates = [
        TraceCandidate("a", 0.2, 0.0),
        TraceCandidate("b", 0.9, 0.0),
    ]

    best = select_best_trace(
        candidates,
        budget=10,
        rng=random.Random(1),
        method="simple_max",
    )

    assert best.trace_id == "b"


def test_select_best_trace_rejects_bad_budget_even_for_simple_max():
    with pytest.raises(ValueError):
        select_best_trace(
            [TraceCandidate("a", 1.0, 0.0)],
            budget=0,
            rng=random.Random(1),
            method="simple_max",
        )


def test_select_best_trace_rejects_unknown_method():
    with pytest.raises(ValueError):
        select_best_trace(
            [TraceCandidate("a", 1.0, 0.0)],
            budget=10,
            rng=random.Random(1),
            method="unknown",
        )


def test_mre_handles_noisy_lucky_outlier_with_sufficient_budget():
    candidates = [
        TraceCandidate("bad_noisy", 0.2, 2.0),
        TraceCandidate("good", 0.9, 0.05),
        TraceCandidate("mid1", 0.5, 0.05),
        TraceCandidate("mid2", 0.45, 0.05),
        TraceCandidate("mid3", 0.4, 0.05),
        TraceCandidate("mid4", 0.35, 0.05),
    ]

    best = multi_round_elimination(candidates, budget=80, rng=random.Random(1234))

    assert best.trace_id == "good"