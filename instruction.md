# Adversarial Trace Selection System

Implement a noise-aware adversarial trace selection system in:

`/app/advnet_selection.py`

This task is inspired by AdvNet's idea of selecting adversarial network traces under noisy evaluation results.

Your module must export:

- `TraceCandidate`
- `evaluate_trace`
- `simple_max`
- `multi_round_elimination`
- `select_best_trace`

Use only the Python standard library.

---

## 1. TraceCandidate

A trace candidate represents one adversarial environment.

```python
TraceCandidate(trace_id: str, true_score: float, noise_std: float = 0.0)
```

Fields:

* `trace_id`: unique candidate identifier
* `true_score`: hidden underlying quality of the candidate
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

* If `noise_std == 0`, return `true_score`.
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
4. Compute each survivor's running mean score.
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
* Candidates with no samples should rank below candidates with samples.
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
* Raise `ValueError` for unknown methods.
* For `"simple_max"`, ignore `budget` except that `budget <= 0` should raise `ValueError`.

---

## Important Edge Cases

Your implementation must correctly handle:

* Empty candidate lists
* Zero noise
* High noise
* Tied observed scores
* Tied running means
* Budgets smaller than number of candidates
* Budgets larger than number of candidates
* One candidate only
* Deterministic behavior with seeded `random.Random`
* Not using global randomness
* Not mutating input candidates

---

## Example

```python
import random
from advnet_selection import TraceCandidate, select_best_trace

candidates = [
    TraceCandidate("a", 0.2, 0.1),
    TraceCandidate("b", 0.8, 0.1),
    TraceCandidate("c", 0.5, 0.1),
]

rng = random.Random(123)
best = select_best_trace(candidates, budget=30, rng=rng)

assert best.trace_id == "b"
```
