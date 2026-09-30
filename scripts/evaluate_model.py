"""How good is the potential model? Scores players it never saw and compares with the truth.

Uses the save (or `--demo` sample) that is already set up. Takes a random sample of the players
the model was NOT taught with, scores them with ONE TabPFN request, and compares with their real
potential. For comparison it also fits scikit-learn gradient boosting (plus two quantile models
for its ranges) on exactly the same 10,000 players. Writes a JSON report to runs/ and the charts
used in the README to docs/.

    python scripts/evaluate_model.py              # 10,000 held-out players, 1 TabPFN call
    python scripts/evaluate_model.py --demo       # the same on the included sample
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from fm26_agent.config import demo_settings, load_settings
from fm26_agent.features import FeatureSchema
from fm26_agent.keys import load_env_file
from fm26_agent.prediction import STAR_LEVEL, HostedPredictor
from fm26_agent.private_db import PrivateStore
from fm26_agent.schema import VISIBLE_ATTRIBUTES
from fm26_agent.visible_db import VisibleStore

AGE_BANDS = (("15-18", 0, 18), ("19-21", 19, 21), ("22-25", 22, 25), ("26+", 26, 99))
CHANCE_BINS = ((0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


# ---------------------------------------------------------------- measures


def errors(actual: np.ndarray, estimate: np.ndarray) -> dict:
    if not len(actual):
        return {"players": 0}
    miss = estimate - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "players": int(len(actual)),
        "mean_abs_error": round(float(np.mean(np.abs(miss))), 2),
        "r2": round(1 - float(np.sum(miss**2)) / total, 3) if total else None,
        "within_5": round(float(np.mean(np.abs(miss) <= 5)), 3),
        "within_10": round(float(np.mean(np.abs(miss) <= 10)), 3),
        "within_15": round(float(np.mean(np.abs(miss) <= 15)), 3),
        "rank_correlation": round(spearman(actual, estimate), 3),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks = [np.argsort(np.argsort(x, kind="stable"), kind="stable") for x in (a, b)]
    return float(np.corrcoef(*ranks)[0, 1])


def ranges(actual: np.ndarray, estimate: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Do the stated 80% ranges hold the real value 80% of the time, and does width mean risk?"""
    inside = (actual >= low) & (actual <= high)
    width, miss = high - low, np.abs(estimate - actual)
    thirds = np.array_split(np.argsort(width, kind="stable"), 3)
    return {
        "claimed_coverage": 0.8,
        "actual_coverage": round(float(inside.mean()), 3),
        "median_width": round(float(np.median(width)), 1),
        "by_width": [
            {
                "group": name,
                "median_width": round(float(np.median(width[idx])), 1),
                "mean_abs_error": round(float(miss[idx].mean()), 2),
                "actual_coverage": round(float(inside[idx].mean()), 3),
            }
            for name, idx in zip(
                ("narrowest third", "middle third", "widest third"), thirds, strict=True
            )
        ],
    }


def auc(positive: np.ndarray, score: np.ndarray) -> float | None:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return round(float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 3)


def chances(actual: np.ndarray, chance: np.ndarray) -> dict:
    """Is "x% chance of reaching 160+" honest? Grouped by what was said, against what happened."""
    star = actual >= STAR_LEVEL
    base = float(star.mean())
    table = []
    for low, high in CHANCE_BINS:
        pick = (chance >= low) & (chance < high)
        if pick.any():
            table.append(
                {
                    "said": f"{round(low * 100)}-{min(round(high * 100), 100)}%",
                    "players": int(pick.sum()),
                    "average_said": round(float(chance[pick].mean()), 3),
                    "really_reached": round(float(star[pick].mean()), 3),
                }
            )
    return {
        "level": STAR_LEVEL,
        "share_who_reached_it": round(base, 4),
        "auc": auc(star, chance),
        "brier_score": round(float(np.mean((chance - star) ** 2)), 4),
        "brier_if_always_saying_the_base_rate": round(base * (1 - base), 4),
        "reliability": table,
    }


def top_picks(actual: np.ndarray, score: np.ndarray, k: int) -> dict:
    top = actual[np.argsort(-score, kind="stable")[:k]]
    return {
        "k": k,
        "mean_real_pa": round(float(top.mean()), 1),
        "reached_160": int((top >= STAR_LEVEL).sum()),
        "best_possible_mean": round(float(np.sort(actual)[-k:].mean()), 1),
    }


# ---------------------------------------------------------------- baseline


NUMERIC = ("age", "height_cm", "value_eur", "wage_eur", "contract_days_remaining", "on_loan")


def standard_table(players: list[dict], counts: dict[str, dict]) -> np.ndarray:
    """What a typical hand-built pipeline feeds boosting: numbers, frequency-encoded categories."""
    rows = []
    for player in players:
        row = [np.nan if player.get(k) is None else float(player[k]) for k in NUMERIC]
        row += [np.nan if player.get(k) is None else float(player[k]) for k in VISIBLE_ATTRIBUTES]
        values = FeatureSchema.categorical_values(player)
        row += [counts[k].get(values[k], 0) for k in counts]
        row += [len(player["traits"]), float(player["preferred_foot"] == "left")]
        rows.append(row)
    return np.array(rows, dtype=float)


def boosting(train: list[dict], y: np.ndarray, test: list[dict]):
    counts: dict[str, dict] = {
        key: {}
        for key in (
            "club_category",
            "nation_category",
            "natural_position_category",
            "accomplished_position_category",
        )
    }
    for player in train:
        for key, value in FeatureSchema.categorical_values(player).items():
            if key in counts:
                counts[key][value] = counts[key].get(value, 0) + 1
    x_train, x_test = standard_table(train, counts), standard_table(test, counts)

    def fit(**kwargs):
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, early_stopping=True, random_state=42, **kwargs
        ).fit(x_train, y)

    point = fit().predict(x_test)
    low = fit(loss="quantile", quantile=0.1).predict(x_test)
    high = fit(loss="quantile", quantile=0.9).predict(x_test)
    return point, low, high


# ---------------------------------------------------------------- charts


INK, MUTED, TABPFN, OTHER, GRID = "#1f2430", "#6b7280", "#2563eb", "#f59e0b", "#e5e7eb"


def _svg(width: int, height: int, body: list[str], title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="13">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<text x="20" y="28" font-size="16" font-weight="bold" fill="{INK}">{title}</text>'
        + "".join(body)
        + "</svg>\n"
    )


def reliability_chart(report: dict) -> str:
    """Said chance vs how often it really happened: points on the diagonal are honest."""
    size, left, top = 300, 60, 50
    rows = report["reliability"]
    body = [
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="{GRID}"/>'
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x, y = left + tick * size, top + (1 - tick) * size
        body += [
            f'<line x1="{left}" y1="{y}" x2="{left + size}" y2="{y}" stroke="{GRID}"/>',
            f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" fill="{MUTED}">{tick:.0%}</text>',
            f'<text x="{x}" y="{top + size + 18}" text-anchor="middle" fill="{MUTED}">{tick:.0%}</text>',
        ]
    body.append(
        f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke="{MUTED}" stroke-dasharray="4 4"/>'
    )
    for row in rows:
        x = left + row["average_said"] * size
        y = top + (1 - row["really_reached"]) * size
        radius = 4 + min(10, np.log10(row["players"] + 1) * 3)
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{TABPFN}" fill-opacity="0.8"/>'
        )
    body += [
        f'<text x="{left + size / 2}" y="{top + size + 40}" text-anchor="middle" fill="{INK}">chance TabPFN gave of reaching {report["level"]}+</text>',
        f'<text transform="translate(18 {top + size / 2}) rotate(-90)" text-anchor="middle" fill="{INK}">share that really did</text>',
        f'<text x="{left + 10}" y="{top + 20}" fill="{MUTED}">AUC {report["auc"]} · dot size = players</text>',
    ]
    return _svg(390, 420, body, "Star chances are honest")


def comparison_chart(report: dict) -> str:
    """TabPFN vs gradient boosting on the same data: error, within 10, range coverage."""
    tab, gbm = report["tabpfn"], report["gradient_boosting"]
    metrics = (
        (
            "Average error (points)",
            tab["mean_abs_error"],
            gbm["mean_abs_error"],
            14,
            "{:.1f}",
            "lower is better",
        ),
        ("Within 10 points", tab["within_10"], gbm["within_10"], 1, "{:.0%}", "higher is better"),
        (
            "80% range holds the truth",
            tab["range_coverage"],
            gbm["range_coverage"],
            1,
            "{:.0%}",
            "80% is right",
        ),
    )
    body, y = [], 60
    for label, a, b, scale, fmt, hint in metrics:
        body.append(f'<text x="20" y="{y}" fill="{INK}" font-weight="bold">{label}</text>')
        body.append(f'<text x="560" y="{y}" text-anchor="end" fill="{MUTED}">{hint}</text>')
        for offset, value, colour, name in (
            (12, a, TABPFN, "TabPFN-3.5"),
            (36, b, OTHER, "Gradient boosting"),
        ):
            width = 360 * value / scale
            body += [
                f'<text x="20" y="{y + offset + 13}" fill="{MUTED}">{name}</text>',
                f'<rect x="150" y="{y + offset}" width="{width:.1f}" height="18" rx="3" fill="{colour}"/>',
                f'<text x="{150 + width + 6:.1f}" y="{y + offset + 13}" fill="{INK}">{fmt.format(value)}</text>',
            ]
        if "range" in label:
            x = 150 + 360 * 0.8
            body.append(
                f'<line x1="{x}" y1="{y + 6}" x2="{x}" y2="{y + 58}" stroke="{INK}" stroke-dasharray="3 3"/>'
            )
        y += 86
    return _svg(
        580,
        y,
        body,
        f"Same {report['taught_players']:,} training players, {report['scored_players']:,} unseen",
    )


def width_chart(report: dict) -> str:
    """Wider range means bigger real miss: the range is a usable risk signal."""
    groups = report["by_width"]
    body, top, height = [], 50, 180
    peak = max(group["mean_abs_error"] for group in groups) * 1.2
    for index, group in enumerate(groups):
        x = 40 + index * 180
        bar = height * group["mean_abs_error"] / peak
        body += [
            f'<rect x="{x + 25}" y="{top + height - bar:.1f}" width="100" height="{bar:.1f}" rx="3" fill="{TABPFN}"/>',
            f'<text x="{x + 75}" y="{top + height - bar - 6:.1f}" text-anchor="middle" fill="{INK}">avg miss {group["mean_abs_error"]:.1f}</text>',
            f'<text x="{x + 75}" y="{top + height + 18}" text-anchor="middle" fill="{INK}">{group["group"]}</text>',
            f'<text x="{x + 75}" y="{top + height + 34}" text-anchor="middle" fill="{MUTED}">~{group["median_width"]:.0f} pts wide</text>',
            f'<text x="{x + 75}" y="{top + height + 50}" text-anchor="middle" fill="{MUTED}">holds truth {group["actual_coverage"]:.0%}</text>',
        ]
    return _svg(600, top + height + 66, body, "Wider range, bigger real miss")


def draw(report: dict, charts: Path) -> None:
    charts.mkdir(parents=True, exist_ok=True)
    (charts / "star-chances.svg").write_text(
        reliability_chart(report["star_chances"]), encoding="utf-8"
    )
    (charts / "tabpfn-vs-boosting.svg").write_text(
        comparison_chart(report["comparison"]), encoding="utf-8"
    )
    (charts / "range-width.svg").write_text(width_chart(report["ranges"]), encoding="utf-8")
    (charts / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=10_000, help="held-out players to score")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--demo", action="store_true", help="evaluate the included sample setup")
    parser.add_argument("--charts", default="docs", help="where to write the SVG charts")
    parser.add_argument(
        "--redraw", action="store_true", help="only redraw the charts from the saved results.json"
    )
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    if args.redraw:
        charts = Path(args.charts)
        draw(json.loads((charts / "results.json").read_text(encoding="utf-8")), charts)
        print(f"Redrew the charts in {charts}/")
        return 0

    settings = load_settings(args.config)
    if args.demo:
        settings = demo_settings(settings)
    load_env_file(settings.project_root)
    if not settings.tabpfn_token:
        print("TABPFN_TOKEN is missing (put it in .env).", file=sys.stderr)
        return 1
    visible = VisibleStore(settings.data.visible_database)
    private = PrivateStore(settings.data.private_database)
    metadata = visible.metadata() if settings.data.visible_database.exists() else {}
    if not metadata.get("model_ready"):
        print("Set the tool up first: run fm26-agent (or fm26-agent --demo).", file=sys.stderr)
        return 1
    predictor = HostedPredictor.load(
        settings.data.model_reference, settings.data.feature_schema, metadata["preparation_id"]
    )
    taught = private.rows("train")
    held_out = [row for row in private.rows("test") if row["potential_ability"] is not None]
    sample = random.Random(args.seed).sample(held_out, min(args.n, len(held_out)))
    players = visible.get_players([row["player_id"] for row in sample])
    by_id = {row["player_id"]: row for row in players}
    ordered = [by_id[row["player_id"]] for row in sample]
    print(f"Taught with {len(taught):,} players; scoring {len(sample):,} it never saw...")

    started = time.monotonic()
    scored = predictor.predict(ordered)  # one hosted request
    seconds = time.monotonic() - started
    actual = np.array([row["potential_ability"] for row in sample], dtype=float)
    estimate = np.array([row["predicted_potential"] for row in scored])
    low = np.array([row["potential_low"] for row in scored])
    high = np.array([row["potential_high"] for row in scored])
    chance = np.array([row["star_chance"] for row in scored])
    age = np.array([row["age"] or 0 for row in ordered])
    known = np.array([row["value_eur"] is not None for row in ordered])
    young = age <= 21

    print("Fitting gradient boosting on the same players for comparison...")
    train_players = visible.get_players([row["player_id"] for row in taught])
    y_train = np.array(
        [
            {r["player_id"]: r for r in taught}[p["player_id"]]["potential_ability"]
            for p in train_players
        ],
        dtype=float,
    )
    g_point, g_low, g_high = boosting(train_players, y_train, ordered)

    report = {
        "source": "sample" if args.demo else metadata.get("source"),
        "taught_players": len(taught),
        "scored_players": len(sample),
        "tabpfn_requests": 1,
        "tabpfn_seconds": round(seconds, 1),
        "guessing_the_average_error": round(
            float(np.mean(np.abs(actual - np.mean([r["potential_ability"] for r in taught])))), 2
        ),
        "overall": errors(actual, estimate),
        "by_age": {
            name: errors(actual[(age >= lo) & (age <= hi)], estimate[(age >= lo) & (age <= hi)])
            for name, lo, hi in AGE_BANDS
        },
        "by_market_value": {
            "has_value": errors(actual[known], estimate[known]),
            "no_value_in_save": errors(actual[~known], estimate[~known]),
        },
        "ranges": ranges(actual, estimate, low, high),
        "star_chances": chances(actual, chance),
        "top_picks": {
            "all_ages_top_25": top_picks(actual, estimate, 25),
            "under_22_top_10": top_picks(actual[young], estimate[young], 10),
            "under_22_top_10_by_star_chance": top_picks(actual[young], chance[young], 10),
        },
        "comparison": {
            "taught_players": len(taught),
            "scored_players": len(sample),
            "tabpfn": errors(actual, estimate)
            | {"range_coverage": round(float(np.mean((actual >= low) & (actual <= high))), 3)},
            "gradient_boosting": errors(actual, g_point)
            | {"range_coverage": round(float(np.mean((actual >= g_low) & (actual <= g_high))), 3)},
        },
    }
    settings.data.runs_directory.mkdir(parents=True, exist_ok=True)
    path = settings.data.runs_directory / f"eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    charts = Path(args.charts)
    charts.mkdir(parents=True, exist_ok=True)
    (charts / "star-chances.svg").write_text(
        reliability_chart(report["star_chances"]), encoding="utf-8"
    )
    (charts / "tabpfn-vs-boosting.svg").write_text(
        comparison_chart(report["comparison"]), encoding="utf-8"
    )
    (charts / "range-width.svg").write_text(width_chart(report["ranges"]), encoding="utf-8")
    (charts / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {path.relative_to(settings.project_root)} and charts to {charts}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
