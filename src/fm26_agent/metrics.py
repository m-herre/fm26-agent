from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(
    ranked_ids: Sequence[int],
    ground_truth: dict[int, dict[str, Any]],
    eligible_ids: Sequence[int],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    total_positives = sum(int(ground_truth[player_id]["wonderkid"]) for player_id in eligible_ids)
    result: dict[str, Any] = {
        "eligible_count": len(eligible_ids),
        "positive_count": total_positives,
        "recommended_count": len(ranked_ids),
    }
    for k in ks:
        selected = list(ranked_ids[:k])
        hits = sum(int(ground_truth[player_id]["wonderkid"]) for player_id in selected)
        result[f"precision_at_{k}"] = hits / len(selected) if selected else None
        result[f"recall_at_{k}"] = hits / total_positives if total_positives else None
        result[f"true_wonderkids_top_{k}"] = hits
        result[f"average_hidden_pa_top_{k}"] = (
            float(np.mean([ground_truth[player_id]["potential_ability"] for player_id in selected]))
            if selected
            else None
        )
    return result


def prediction_metrics(
    predictions: Sequence[dict[str, Any]], ground_truth: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    ids = [row["player_id"] for row in predictions]
    targets = [ground_truth[player_id]["wonderkid"] for player_id in ids]
    scores = [row["wonderkid_probability"] for row in predictions]
    ranked = sorted(predictions, key=lambda row: (-row["wonderkid_probability"], row["player_id"]))
    result = ranking_metrics([row["player_id"] for row in ranked], ground_truth, ids)
    both_classes = len(set(targets)) == 2
    result["roc_auc"] = float(roc_auc_score(targets, scores)) if both_classes else None
    result["average_precision"] = (
        float(average_precision_score(targets, scores)) if both_classes else None
    )
    result["positive_count"] = sum(targets)
    return result


def regression_metrics(predictions, ground_truth):
    """Offline ground-truth evaluation only; never imported by scouting tools."""
    from sklearn.metrics import r2_score

    ids = [row["player_id"] for row in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate regression prediction IDs")
    ranked = sorted(predictions, key=lambda row: (-row["predicted_potential"], row["player_id"]))
    result = ranking_metrics([row["player_id"] for row in ranked], ground_truth, ids)
    actual = np.asarray([ground_truth[i]["potential_ability"] for i in ids], dtype=float)
    estimated = np.asarray([row["predicted_potential"] for row in predictions], dtype=float)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(estimated)):
        raise ValueError("Regression metrics require finite predictions and exact targets")
    result["mae"] = float(np.mean(np.abs(actual - estimated))) if ids else None
    result["rmse"] = float(np.sqrt(np.mean((actual - estimated) ** 2))) if ids else None
    result["r2"] = (
        float(r2_score(actual, estimated)) if len(ids) >= 2 and np.ptp(actual) > 0 else None
    )
    return result
