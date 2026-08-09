import math
import random

def boltzmann(scores: dict, temperature: float) -> dict:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {k: math.exp((v - max_score) / temperature) for k, v in scores.items()}
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}

def sample(distribution: dict, n: int = 1) -> list:
    if not distribution:
        return []

    items = []
    dist = distribution.copy()

    for _ in range(min(n, len(dist))):
        keys = list(dist.keys())
        weights = list(dist.values())

        chosen = random.choices(keys, weights=weights, k=1)[0]
        items.append(chosen)

        del dist[chosen]

    return items