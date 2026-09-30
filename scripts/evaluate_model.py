"""Developer check: how well does the saved potential model do on players it never saw?

Not part of the shipped tool. Uses the save that is already set up (`fm26-agent` first), scores a
random sample of the players the model was NOT taught with, and compares with their real
potential. Needs the TabPFN key in `.env`. Writes a JSON report to runs/.

    python scripts/evaluate_model.py            # 3,000 held-out players
    python scripts/evaluate_model.py --n 10000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime

import numpy as np

from fm26_agent.config import load_settings
from fm26_agent.keys import load_env_file
from fm26_agent.prediction import HostedPredictor
from fm26_agent.private_db import PrivateStore
from fm26_agent.visible_db import VisibleStore

WONDERKID = 160
AGE_BANDS = (("15-18", 0, 18), ("19-21", 19, 21), ("22-25", 22, 25), ("26+", 26, 99))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        result = np.empty(len(x))
        result[order] = np.arange(len(x))
        return result

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1]) if len(a) > 2 else float("nan")


def errors(actual: np.ndarray, estimate: np.ndarray, baseline: float) -> dict:
    if not len(actual):
        return {"players": 0}
    miss = estimate - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "players": int(len(actual)),
        "mean_abs_error": round(float(np.mean(np.abs(miss))), 2),
        "rmse": round(float(np.sqrt(np.mean(miss**2))), 2),
        "r2": round(1 - float(np.sum(miss**2)) / total, 3) if total else None,
        "bias": round(float(np.mean(miss)), 2),
        "within_5": round(float(np.mean(np.abs(miss) <= 5)), 3),
        "within_10": round(float(np.mean(np.abs(miss) <= 10)), 3),
        "within_15": round(float(np.mean(np.abs(miss) <= 15)), 3),
        "spearman": round(spearman(actual, estimate), 3),
        "mean_abs_error_if_always_guessing_the_average": round(
            float(np.mean(np.abs(actual - baseline))), 2
        ),
    }


def ranking(actual: np.ndarray, estimate: np.ndarray) -> dict:
    """If you looked at the model's top K, how many are real wonderkids?"""
    wonderkids = actual >= WONDERKID
    order = np.argsort(-estimate, kind="stable")
    result = {"wonderkids_in_group": int(wonderkids.sum())}
    for k in (10, 25, 50, 100):
        if len(actual) >= k:
            top = order[:k]
            result[f"top_{k}"] = {
                "real_wonderkids": int(wonderkids[top].sum()),
                "mean_real_pa": round(float(actual[top].mean()), 1),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=3000, help="held-out players to score")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    settings = load_settings(args.config)
    load_env_file(settings.project_root)
    if not settings.tabpfn_token:
        print("TABPFN_TOKEN is missing (put it in .env).", file=sys.stderr)
        return 1
    visible = VisibleStore(settings.data.visible_database)
    private = PrivateStore(settings.data.private_database)
    metadata = visible.metadata()
    if not metadata.get("model_ready"):
        print("Set the tool up first: run fm26-agent.", file=sys.stderr)
        return 1
    predictor = HostedPredictor.load(
        settings.data.model_reference, settings.data.feature_schema, metadata["preparation_id"]
    )
    taught = private.rows("train")
    held_out = [row for row in private.rows("test") if row["potential_ability"] is not None]
    rng = random.Random(args.seed)
    sample = rng.sample(held_out, min(args.n, len(held_out)))
    print(
        f"Taught with {len(taught):,} players; scoring {len(sample):,} of "
        f"{len(held_out):,} others it never saw..."
    )
    players = visible.get_players([row["player_id"] for row in sample])
    by_id = {row["player_id"]: row for row in players}
    scored = {}
    for start in range(0, len(sample), 1000):
        chunk = [by_id[row["player_id"]] for row in sample[start : start + 1000]]
        scored.update({r["player_id"]: r["predicted_potential"] for r in predictor.predict(chunk)})
        print(f"  {min(start + 1000, len(sample)):,}/{len(sample):,}")
    actual = np.array([row["potential_ability"] for row in sample], dtype=float)
    estimate = np.array([scored[row["player_id"]] for row in sample])
    age = np.array([by_id[row["player_id"]]["age"] or 0 for row in sample])
    known = np.array([by_id[row["player_id"]]["value_eur"] is not None for row in sample])
    baseline = float(np.mean([row["potential_ability"] for row in taught]))
    report = {
        "preparation_id": metadata["preparation_id"],
        "taught_players": len(taught),
        "scored_players": len(sample),
        "seed": args.seed,
        "overall": errors(actual, estimate, baseline),
        "by_age": {
            name: errors(
                actual[(age >= low) & (age <= high)],
                estimate[(age >= low) & (age <= high)],
                baseline,
            )
            for name, low, high in AGE_BANDS
        },
        "by_market_value": {
            "has_value": errors(actual[known], estimate[known], baseline),
            "no_value_in_save": errors(actual[~known], estimate[~known], baseline),
        },
        "ranking_all_ages": ranking(actual, estimate),
        "ranking_under_22": ranking(actual[age <= 21], estimate[age <= 21]),
    }
    settings.data.runs_directory.mkdir(parents=True, exist_ok=True)
    path = settings.data.runs_directory / f"eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {path.relative_to(settings.project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
