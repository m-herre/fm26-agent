from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureSchema
from .schema import POSITION_CODES

SAMPLER_VERSION = "target-quotas-marginal-swaps-v2"
MISSING = "[missing]"


def _strata(
    players: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Each column is a marginal partition, never a cross-product of features."""
    partitions: dict[str, list[str]] = {}
    numeric: dict[str, np.ndarray] = {}
    for key in NUMERIC_FEATURES:
        values = pd.to_numeric(
            pd.Series([row.get(key) for row in players]), errors="coerce"
        ).to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        numeric[key] = values
        finite = values[np.isfinite(values)]
        edges = (
            np.unique(np.quantile(finite, np.linspace(0, 1, 11))) if len(finite) else np.array([])
        )
        bins = np.searchsorted(edges[1:-1], values, side="right")
        partitions[key] = [
            MISSING if not np.isfinite(value) else f"bin:{bucket}"
            for value, bucket in zip(values, bins, strict=True)
        ]
    categories = [FeatureSchema.categorical_values(row) for row in players]
    for key in CATEGORICAL_FEATURES:
        partitions[key] = [row[key] if row[key] is not None else MISSING for row in categories]
    for kind in ("natural", "accomplished"):
        for position in POSITION_CODES:
            partitions[f"{kind}_{position}"] = [
                str(position in row.get(f"{kind}_positions", [])) for row in players
            ]
    traits = sorted({trait for row in players for trait in row.get("traits", [])})
    for trait in traits:
        partitions[f"trait:{trait}"] = [str(trait in row.get("traits", [])) for row in players]
    partitions["trait_text_available"] = [str(bool(row.get("traits"))) for row in players]
    columns, metadata, offset = [], {}, 0
    for key, values in partitions.items():
        labels = sorted(set(values))
        lookup = {value: index for index, value in enumerate(labels)}
        codes = np.asarray([lookup[value] for value in values], dtype=np.int32) + offset
        columns.append(codes)
        metadata[key] = {"offset": offset, "labels": labels}
        offset += len(labels)
    return np.column_stack(columns), metadata, numeric


def representative_sample(
    players: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    size: int = 10_000,
    seed: int = 42,
    max_swaps: int = 3000,
) -> tuple[list[int], dict[str, Any]]:
    """Fixed target quotas plus deterministic within-class swaps balancing observable marginals."""
    by_id = {row["player_id"]: row for row in players}
    eligible = sorted(
        [row for row in labels if row["wonderkid"] is not None], key=lambda row: row["player_id"]
    )
    if not 2 <= size < len(eligible):
        raise ValueError(
            f"Need more than {size:,} exactly-labelled players for the fixed reference and held-out evaluation"
        )
    target = np.asarray([row["wonderkid"] for row in eligible], dtype=int)
    if set(target) != {0, 1}:
        raise ValueError("Reference fitting requires both target classes")
    population = [by_id[row["player_id"]] for row in eligible]
    codes, metadata, numeric = _strata(population)
    dimensions = max(item["offset"] + len(item["labels"]) for item in metadata.values())
    population_counts = np.bincount(codes.ravel(), minlength=dimensions).astype(float)
    expected = population_counts * size / len(eligible)
    weights = 1 / np.maximum(expected, 1)
    for item in metadata.values():
        offset, length = item["offset"], len(item["labels"])
        weights[offset : offset + length] /= np.sqrt(length)
    coverage_penalty = np.zeros(dimensions)
    for key, item in metadata.items():
        if key in CATEGORICAL_FEATURES or key.startswith("trait:"):
            offset, length = item["offset"], len(item["labels"])
            rare = expected[offset : offset + length] < 1
            coverage_penalty[offset : offset + length] = np.where(
                rare, 2 * weights[offset : offset + length], 0
            )
    positive_quota = int(np.floor(size * target.mean() + 0.5))
    positive_quota = max(1, min(size - 1, int(target.sum()), positive_quota))
    negative_quota = size - positive_quota
    if negative_quota > int((target == 0).sum()):
        negative_quota = int((target == 0).sum())
        positive_quota = size - negative_quota
    rng = np.random.default_rng(seed)
    selected, excluded = {}, {}
    for cls, quota in ((0, negative_quota), (1, positive_quota)):
        indices = rng.permutation(np.flatnonzero(target == cls))
        selected[cls], excluded[cls] = indices[:quota].copy(), indices[quota:].copy()
    initial = np.concatenate(list(selected.values()))
    counts = np.bincount(codes[initial].ravel(), minlength=dimensions).astype(float)

    def objective():
        return float(np.sum(weights * (counts - expected) ** 2 + coverage_penalty * (counts == 0)))

    before = objective()
    active = [cls for cls in (0, 1) if len(selected[cls]) and len(excluded[cls])]
    swaps, stale = 0, 0
    for _ in range(max_swaps):
        if not active or stale >= 100:
            break
        cls = int(
            rng.choice(
                active,
                p=np.asarray([len(selected[c]) for c in active])
                / sum(len(selected[c]) for c in active),
            )
        )
        out_slots = rng.integers(len(selected[cls]), size=128)
        in_slots = rng.integers(len(excluded[cls]), size=128)
        outgoing, incoming = codes[selected[cls][out_slots]], codes[excluded[cls][in_slots]]
        changed = outgoing != incoming
        residual = counts - expected
        delta = np.sum(
            np.where(
                changed,
                weights[outgoing] * (1 - 2 * residual[outgoing])
                + weights[incoming] * (1 + 2 * residual[incoming]),
                0,
            ),
            axis=1,
        )
        delta += np.sum(
            np.where(
                changed,
                coverage_penalty[outgoing] * (counts[outgoing] == 1)
                - coverage_penalty[incoming] * (counts[incoming] == 0),
                0,
            ),
            axis=1,
        )
        best = int(np.argmin(delta))
        if delta[best] >= -1e-10:
            stale += 1
            continue
        stale = 0
        out_slot, in_slot = out_slots[best], in_slots[best]
        out_index, in_index = selected[cls][out_slot], excluded[cls][in_slot]
        mask = changed[best]
        counts[outgoing[best][mask]] -= 1
        counts[incoming[best][mask]] += 1
        selected[cls][out_slot], excluded[cls][in_slot] = in_index, out_index
        swaps += 1
    chosen = np.sort(np.concatenate(list(selected.values())))
    report: dict[str, Any] = {
        "sampler_version": SAMPLER_VERSION,
        "seed": seed,
        "population_rows": len(eligible),
        "reference_rows": len(chosen),
        "population_positive_fraction": float(target.mean()),
        "reference_positive_fraction": float(target[chosen].mean()),
        "positive_quota": positive_quota,
        "accepted_swaps": swaps,
        "objective_before": before,
        "objective_after": objective(),
        "objective": "Weighted marginal count error plus a rare-category coverage penalty; swaps never change target quotas. Approximate, not exact, feature balance. Rare coverage may trade off against frequency matching.",
        "features": {},
        "scouting_notes_available": False,
    }
    for key, item in metadata.items():
        offset, labels_for_feature = item["offset"], item["labels"]
        pop = population_counts[offset : offset + len(labels_for_feature)]
        sample = counts[offset : offset + len(labels_for_feature)]
        entry = {
            "total_variation": float(np.abs(pop / len(eligible) - sample / size).sum() / 2),
            "population_categories": len(pop),
            "covered_categories": int((sample > 0).sum()),
            "category_coverage": float((sample > 0).mean()),
            "covered_population_mass": float(pop[sample > 0].sum() / len(eligible)),
            "rare_categories": int((pop * size / len(eligible) < 1).sum()),
            "covered_rare_categories": int(((pop * size / len(eligible) < 1) & (sample > 0)).sum()),
            "missing_population_fraction": float(
                pop[labels_for_feature.index(MISSING)] / len(eligible)
            )
            if MISSING in labels_for_feature
            else 0.0,
            "missing_reference_fraction": float(sample[labels_for_feature.index(MISSING)] / size)
            if MISSING in labels_for_feature
            else 0.0,
            "distributions": [
                {"group": label, "population_count": int(p), "reference_count": int(s)}
                for label, p, s in zip(labels_for_feature, pop, sample, strict=True)
            ],
        }
        if key in numeric:
            full = np.sort(numeric[key][np.isfinite(numeric[key])])
            subset = np.sort(numeric[key][chosen][np.isfinite(numeric[key][chosen])])
            if len(full) and len(subset):
                domain = np.union1d(full, subset)
                entry["ks_distance"] = float(
                    np.max(
                        np.abs(
                            np.searchsorted(full, domain, side="right") / len(full)
                            - np.searchsorted(subset, domain, side="right") / len(subset)
                        )
                    )
                )
                entry["population_mean"], entry["reference_mean"] = (
                    float(full.mean()),
                    float(subset.mean()),
                )
        report["features"][key] = entry
    return [eligible[index]["player_id"] for index in chosen], report
