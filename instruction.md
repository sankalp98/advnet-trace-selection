# Adversarial Trace Selection System

Implement a noise-aware adversarial trace selection system in:

`/app/advnet_selection.py`

This task is inspired by AdvNet’s idea of selecting adversarial network traces under noisy evaluation results.

Your module must export:

- `TraceCandidate`
- `evaluate_trace`
- `simple_max`
- `multi_round_elimination`
- `select_best_trace`

Use only the Python standard library.

---

## Background

An adversarial testing system evaluates many candidate network traces.

Each trace has an underlying quality score, but evaluations are noisy. A trace may look good because it is truly strong, or because it received a lucky noisy sample.

The goal is to select the candidate with the best estimated quality while respecting a limited evaluation budget.

---

## 1. TraceCandidate

A trace candidate represents one adversarial environment.

```python
TraceCandidate(trace_id: str, true_score: float, noise_std: float = 0.0)
````

Fields:

* `trace_id`: unique candidate identifier
* `true_score`: underlying quality of the candidate
* `noise_std`: standard deviation of Gaussian evaluation noise

Rules:

* `trace_id` must be a non-empty string.
* `true_score` must be a finite number.
* `noise_std` must be a finite non-negative number.

You may implement this as a dataclass.

---

## 2. evaluate_trace

```python
def evaluate_trace(candidate: TraceCandidate, rng: random.Random) -> float:
```

Return one noisy evaluation sample:

```text
sample = candidate.true_score + gaussian_noise
```

where:

```python
gaussian_noise = rng.gauss(0.0, candidate.noise_std)
```

Rules:

* Use the provided `rng`.
* Do not use global randomness.

---

## 3. simple_max

```python
def simple_max(
    candidates: list[TraceCandidate],
    rng: random.Random
) -> TraceCandidate:
```

Evaluate each candidate exactly once and return the candidate with the highest observed sample.

Rules:

* Raise `ValueError` if `candidates` is empty.
* Use `evaluate_trace`.
* If observed scores tie, return the candidate with lexicographically smallest `trace_id`.
* Do not mutate the input list.

---

## 4. multi_round_elimination

```python
def multi_round_elimination(
    candidates: list[TraceCandidate],
    budget: int,
    rng: random.Random
) -> TraceCandidate:
```

Select a candidate robustly under noisy evaluations.

The algorithm:

1. Start with all candidates as survivors.
2. Each survivor has a list of observed samples.
3. In each elimination round, evaluate every survivor once.
4. Compute each survivor’s running mean score.
5. If more than 5 survivors remain, eliminate the bottom half by running mean.
6. Keep the top `ceil(n / 2)` survivors.
7. Tie-break by lexicographically smaller `trace_id`.
8. Once 5 or fewer survivors remain, spend the remaining budget in round-robin order across survivors.
9. Return the survivor with the highest final running mean.
10. Final tie-break is lexicographically smallest `trace_id`.

Rules:

* Raise `ValueError` if `candidates` is empty.
* Raise `ValueError` if `budget <= 0`.
* Never perform more than `budget` total evaluations.
* If the budget is too small to evaluate every survivor in a full round, evaluate candidates in lexicographic `trace_id` order until the budget is exhausted, then return the best candidate by available running mean.
* Candidates with no samples rank below candidates with samples.
* Do not mutate the input list.
* The function must be deterministic for the same `rng` seed.

---

## 5. select_best_trace

```python
def select_best_trace(
    candidates: list[TraceCandidate],
    budget: int,
    rng: random.Random,
    method: str = "mre"
) -> TraceCandidate:
```

Select the best trace.

Supported methods:

* `"mre"`: use `multi_round_elimination`
* `"simple_max"`: use `simple_max`

Rules:

* Default method must be `"mre"`.
* Raise `ValueError` if `budget <= 0`.
* Raise `ValueError` for unknown methods.
* For `"simple_max"`, the budget does not change how many candidates are evaluated.

---

## Correctness Requirements

Your implementation must follow the exact algorithmic behavior described above.

Additional requirements:

* Use only the provided `random.Random` instance for randomness.
* Do not use module-level or global randomness.
* Do not mutate the input candidate list.
* Candidate ordering and tie-breaking must be deterministic.
* Invalid inputs must raise `ValueError` where specified.
* The implementation must never exceed the provided evaluation budget.