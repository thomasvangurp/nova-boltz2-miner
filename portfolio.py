"""Build the highest-scoring portfolio that the validator will accept."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from rdkit import DataStructs


class CandidateLike(Protocol):
    name: str
    morgan: object
    maccs: np.ndarray
    observations: list[float]

    def conservative_score(self, noise_floor: float) -> float: ...


def maccs_entropy(rows: Sequence[np.ndarray]) -> float:
    """Match the validator's MACCS entropy calculation."""
    matrix = np.asarray(rows, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[1] != 167 or not len(matrix):
        raise ValueError("MACCS entropy needs non-empty 167-bit rows")
    probabilities = matrix.astype(np.float64).sum(axis=0) / len(matrix)
    entropy = [
        -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p) if 0.0 < p < 1.0 else 0.0
        for p in probabilities
    ]
    return float(np.mean(entropy))


def _entropy_after(total: np.ndarray, row: np.ndarray, count: int) -> float:
    probabilities = (total + row) / count
    active = (probabilities > 0.0) & (probabilities < 1.0)
    entropy = np.zeros(167, dtype=np.float64)
    p = probabilities[active]
    entropy[active] = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return float(entropy.mean())


def _is_diverse(
    candidate: CandidateLike, selected: Sequence[CandidateLike], threshold: float
) -> bool:
    if not selected:
        return True
    similarities = DataStructs.BulkTanimotoSimilarity(
        candidate.morgan, [item.morgan for item in selected]
    )
    return max(similarities, default=0.0) < threshold


def _greedy_select(
    candidates: Sequence[CandidateLike],
    scores: dict[str, float],
    count: int,
    tanimoto_threshold: float,
    entropy_weight: float,
) -> list[CandidateLike]:
    """Greedy score/entropy selection with the exact pairwise diversity gate."""
    remaining = list(candidates)
    selected: list[CandidateLike] = []
    bit_sum = np.zeros(167, dtype=np.float64)

    while remaining and len(selected) < count:
        best_index = None
        best_value = -math.inf
        next_count = len(selected) + 1
        for index, candidate in enumerate(remaining):
            if not _is_diverse(candidate, selected, tanimoto_threshold):
                continue
            entropy = _entropy_after(bit_sum, candidate.maccs, next_count)
            value = scores[candidate.name] + entropy_weight * entropy
            if value > best_value:
                best_index = index
                best_value = value
        if best_index is None:
            break
        best = remaining.pop(best_index)
        selected.append(best)
        bit_sum += best.maccs
    return selected


def select_portfolio(
    candidates: Sequence[CandidateLike],
    count: int,
    tanimoto_threshold: float,
    entropy_threshold: float,
    noise_floor: float,
    pool_limit: int = 900,
) -> tuple[list[CandidateLike], float, float]:
    """Return an exact-size, diverse, entropy-safe portfolio.

    First try the pure score ordering. Entropy-biased alternatives are only
    evaluated when that fastest and highest-scoring path misses the floor.
    """
    if len(candidates) < count:
        return [], -math.inf, 0.0

    scored = [candidate for candidate in candidates if candidate.observations]
    score_floor = (
        min(
            (candidate.conservative_score(noise_floor) for candidate in scored),
            default=0.0,
        )
        - 1.0
    )
    scores = {
        candidate.name: (
            candidate.conservative_score(noise_floor)
            if candidate.observations
            else score_floor
        )
        for candidate in candidates
    }
    ranked = sorted(candidates, key=lambda item: scores[item.name], reverse=True)[
        :pool_limit
    ]

    def evaluate(entropy_weight: float):
        chosen = _greedy_select(
            ranked, scores, count, tanimoto_threshold, entropy_weight
        )
        if len(chosen) != count:
            return None
        entropy = maccs_entropy([candidate.maccs for candidate in chosen])
        mean_score = float(np.mean([scores[candidate.name] for candidate in chosen]))
        return chosen, mean_score, entropy

    direct = evaluate(0.0)
    if direct is not None and direct[2] > entropy_threshold:
        return direct

    score_scale = max(
        0.005,
        float(np.std([scores[candidate.name] for candidate in scored]))
        if len(scored) > 1
        else 0.0,
    )
    feasible = []
    best_any = direct
    for multiplier in (0.5, 1, 2, 4, 8, 16, 32):
        result = evaluate(score_scale * multiplier)
        if result is None:
            continue
        if best_any is None or (result[2], result[1]) > (best_any[2], best_any[1]):
            best_any = result
        if result[2] > entropy_threshold:
            feasible.append(result)
    if feasible:
        return max(feasible, key=lambda result: result[1])
    return best_any if best_any is not None else ([], -math.inf, 0.0)


def write_result(
    path: str | os.PathLike[str], portfolio: Sequence[CandidateLike]
) -> None:
    """Atomically replace result.json so a timeout cannot leave partial JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {"molecules": [candidate.name for candidate in portfolio]},
                handle,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
