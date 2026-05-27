# Adversarial Trace Selection

Implement a noise-aware trace selection module at:

`/app/advnet_selection.py`

Use only the Python standard library.

This task is inspired by AdvNet’s post-learning selection problem: many candidate network traces are evaluated under noise, and you must pick the best one within a limited evaluation budget.

---

## Exports

Your module must export:

- `TraceCandidate`
- `evaluate_trace`
- `simple_max`
- `multi_round_elimination`
- `select_best_trace`

---

## API

### `TraceCandidate`

```python
TraceCandidate(trace_id: str, true_score: float, noise_std: float = 0.0)
```

Represents one candidate adversarial trace.

- `trace_id`: unique identifier (non-empty string)
- `true_score`: underlying quality of the candidate
- `noise_std`: scale of evaluation noise

Reject invalid inputs with `ValueError`.

### `evaluate_trace`

```python
def evaluate_trace(candidate: TraceCandidate, rng: random.Random) -> float:
```

Return one stochastic evaluation of `candidate`.

- Use only the provided `rng` (no global or module-level randomness).

### `simple_max`

```python
def simple_max(candidates: list[TraceCandidate], rng: random.Random) -> TraceCandidate:
```

Baseline selector: choose the candidate that looks best from a single noisy evaluation per candidate.

- Raise `ValueError` if `candidates` is empty.
- Do not mutate the input list.

### `multi_round_elimination`

```python
def multi_round_elimination(
    candidates: list[TraceCandidate],
    budget: int,
    rng: random.Random,
) -> TraceCandidate:
```

Robust selector under noise and a fixed evaluation budget.

Repeated sampling should matter more than one lucky draw. This is the main selection strategy (AdvNet-style MRE / post-learning selection).

- Raise `ValueError` if `candidates` is empty or `budget <= 0`.
- Never perform more than `budget` total evaluations.
- Do not mutate the input list.
- Behavior must be deterministic for the same `rng` seed.

### `select_best_trace`

```python
def select_best_trace(
    candidates: list[TraceCandidate],
    budget: int,
    rng: random.Random,
    method: str = "mre",
) -> TraceCandidate:
```

Dispatch to a selection method:

- `"mre"` → `multi_round_elimination` (default)
- `"simple_max"` → `simple_max`

Raise `ValueError` for unknown methods or `budget <= 0`.

---

## Requirements

- Stdlib only.
- Deterministic for a fixed `rng` seed.
- Do not mutate input candidate lists.
- Correctness is defined by the test suite.
