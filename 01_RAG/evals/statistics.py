from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> ConfidenceInterval:
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    if not numeric:
        raise ValueError("bootstrap 至少需要一个有效样本")
    _validate_parameters(confidence, samples)
    estimate = sum(numeric) / len(numeric)
    rng = random.Random(seed)
    distribution = sorted(
        sum(rng.choice(numeric) for _ in numeric) / len(numeric)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=estimate,
        lower=_quantile(distribution, tail),
        upper=_quantile(distribution, 1.0 - tail),
    )


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> ConfidenceInterval:
    if len(baseline) != len(candidate):
        raise ValueError("Baseline 与 Candidate 必须包含相同 Case")
    if not baseline:
        raise ValueError("配对 bootstrap 至少需要一个 Case")
    pairs = [
        (float(left), float(right))
        for left, right in zip(baseline, candidate)
    ]
    if any(not math.isfinite(left) or not math.isfinite(right) for left, right in pairs):
        raise ValueError("配对 bootstrap 只接受有限数值")
    _validate_parameters(confidence, samples)

    deltas = [right - left for left, right in pairs]
    estimate = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    distribution = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=estimate,
        lower=_quantile(distribution, tail),
        upper=_quantile(distribution, 1.0 - tail),
    )


def _validate_parameters(confidence: float, samples: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于 0 和 1 之间")
    if samples <= 0:
        raise ValueError("samples 必须大于 0")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return (
        float(sorted_values[lower_index]) * (1.0 - fraction)
        + float(sorted_values[upper_index]) * fraction
    )
