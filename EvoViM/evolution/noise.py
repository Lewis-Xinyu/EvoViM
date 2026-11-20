"""Parent repetition noise and minimum-detectable-effect calculations."""
from __future__ import annotations

import math
import statistics
from statistics import NormalDist


def estimate_parent_noise(scores_by_seed: dict[int | str, list[float]], z: float | None = None,
                          *, alpha: float = 0.05, power: float = 0.80) -> dict:
    variances = {}
    for seed, scores in scores_by_seed.items():
        if len(scores) < 2:
            raise ValueError("parent_repetitions must be at least two")
        variances[str(seed)] = statistics.variance(scores)
    mean_variance = statistics.mean(variances.values()) if variances else 0.0
    repetitions = min(len(scores) for scores in scores_by_seed.values())
    seed_count = len(scores_by_seed)
    z_alpha = (NormalDist().inv_cdf(1 - alpha / 2)) if z is None else 1.96
    z_beta = NormalDist().inv_cdf(power)
    mde = ((z_alpha + z_beta) * math.sqrt(2 * mean_variance / (repetitions * seed_count))
           if repetitions and seed_count else float("inf"))
    return {"per_seed_variance": variances, "mean_variance": mean_variance, "repetitions": repetitions,
            "seed_count": seed_count, "alpha": alpha, "power": power,
            "z_alpha": z_alpha, "z_beta": z_beta, "mde_80pct": mde}
