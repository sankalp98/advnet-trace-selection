import math
import random
import pytest

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
        TraceCandidate("a", 1.0, -0.1)


def test_evaluate_trace_zero_noise_is_exact():
    c = TraceCandidate("x", 0.75, 0.0)
    rng = random.Random(123)

    assert evaluate_trace(c, rng) == 0.75
    assert evaluate_trace(c, rng) == 0.75


def test_evaluate_trace_uses_seeded_rng():
    c = TraceCandidate("x", 1.0, 0.5)

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    assert evaluate_trace(c, rng1) == evaluate_trace(c, rng2)


def test_simple_max_evaluates_once_and_uses_observed_score():
    candidates = [
        TraceCandidate("a", 0.1, 0.0),
        TraceCandidate("b", 0.9, 0.0),
        TraceCandidate("c", 0.5, 0.0),
    ]

    best = simple_max(candidates, random.Random(1))
    assert best.trace_id == "b"


def test_simple_max_tie_breaks_by_trace_id():
    candidates = [
        TraceCandidate("z", 1.0, 0.0),
        TraceCandidate("a", 1.0, 0.0),
        TraceCandidate("m", 1.0, 0.0),
    ]

    best = simple_max(candidates, random.Random(1))
    assert best.trace_id == "a"


def test_simple_max_empty_raises():
    with pytest.raises(ValueError):
        simple_max([], random.Random(1))


def test_mre_selects_best_without_noise():
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


def test_mre_handles_budget_smaller_than_candidates():
    candidates = [
        TraceCandidate("a", 0.7, 0.0),
        TraceCandidate("b", 0.2, 0.0),
        TraceCandidate("c", 0.1, 0.0),
    ]

    best = multi_round_elimination(candidates, budget=1, rng=random.Random(10))
    assert best.trace_id == "a"


def test_mre_uses_running_mean_not_single_last_sample():
    candidates = [
        TraceCandidate("a", 0.55, 0.0),
        TraceCandidate("b", 0.50, 0.0),
        TraceCandidate("c", 0.45, 0.0),
        TraceCandidate("d", 0.40, 0.0),
        TraceCandidate("e", 0.35, 0.0),
        TraceCandidate("f", 0.30, 0.0),
        TraceCandidate("g", 0.25, 0.0),
    ]

    best = multi_round_elimination(candidates, budget=50, rng=random.Random(5))
    assert best.trace_id == "a"


def test_mre_tie_breaks_final_by_trace_id():
    candidates = [
        TraceCandidate("b", 1.0, 0.0),
        TraceCandidate("a", 1.0, 0.0),
        TraceCandidate("c", 1.0, 0.0),
    ]

    best = multi_round_elimination(candidates, budget=10, rng=random.Random(1))
    assert best.trace_id == "a"


def test_mre_is_deterministic_for_same_seed():
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
    original_order = [c.trace_id for c in candidates]

    multi_round_elimination(candidates, budget=10, rng=random.Random(1))

    assert [c.trace_id for c in candidates] == original_order


def test_mre_empty_and_bad_budget_raise():
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

    best = select_best_trace(candidates, budget=10, rng=random.Random(1), method="simple_max")
    assert best.trace_id == "b"


def test_select_best_trace_unknown_method_raises():
    with pytest.raises(ValueError):
        select_best_trace(
            [TraceCandidate("a", 1.0)],
            budget=10,
            rng=random.Random(1),
            method="unknown",
        )


def test_noisy_mre_usually_beats_lucky_low_quality_candidate():
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
