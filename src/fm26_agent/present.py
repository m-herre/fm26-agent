"""How objectives and their results look in the terminal (and, as text, anywhere else).

Every number is printed by code from the objective, the predictions and the database; the LLM only
supplies the short explanations and an optional note.
"""

from __future__ import annotations

from typing import Any

from .agent import _amount, format_season_stats, price
from .objective import Condition, Objective, ObjectiveResult
from .targets import get_target
from .visible_db import VisibleStore

FILTER_WORDS = {
    "position": "position {}",
    "age_min": "age {}+",
    "age_max": "age up to {}",
    "value_min_eur": "price from {}",
    "value_max_eur": "price up to {}",
    "club": "club {}",
    "preferred_foot": "{}-footed",
    "contract_ends_within_days": "contract ends within {} days",
    "similar_to": "profile like player {}",
    "height_min_cm": "at least {} cm tall",
    "height_max_cm": "at most {} cm tall",
}


def _chance(value: float) -> str:
    percent = round(value * 100)
    return "over 95%" if percent > 95 else "under 5%" if percent < 5 else f"{percent}%"


def describe_filters(filters: dict[str, Any]) -> str:
    if filters.get("age_min") is not None and filters.get("age_min") == filters.get("age_max"):
        filters = {k: v for k, v in filters.items() if k not in ("age_min", "age_max")} | {
            "age": filters["age_min"]
        }
    parts = []
    for key, value in filters.items():
        if value is None:
            continue
        if key == "age":
            parts.append(f"age {value}")
        elif key in ("value_min_eur", "value_max_eur"):
            parts.append(FILTER_WORDS[key].format(_amount(value)))
        elif key == "club" and isinstance(value, list):
            parts.append("club " + " or ".join(value))
        elif key == "position" and isinstance(value, list):
            parts.append("position " + " or ".join(value))
        elif key == "min_attributes":
            parts.extend(
                f"{name.replace('_', ' ')} {minimum:g}+" for name, minimum in value.items()
            )
        else:
            parts.append(FILTER_WORDS[key].format(value))
    return " · ".join(parts) or "everyone"


def quality_line(target: str, quality: dict[str, Any]) -> str:
    label = get_target(target).label
    verdict = quality.get("verdict", "unchecked")
    if target == "price_vs_fair_value":
        return (
            f"{label}: fair value typically {quality.get('typical_error_percent', '?'):g}% off on "
            f"players whose price it didn't see ({verdict})"
        )
    if target == "price_vs_peers":
        return f"{label}: a comparison with {quality.get('peers', 'similar players')}"
    if "average_error" in quality:
        return (
            f"{label}: off by {quality['average_error']:g} on average on unseen players, "
            f"guessing would be {quality['average_error_if_guessing']:g} ({verdict})"
        )
    return f"{label}: {verdict}"


def render_objective(
    objective: Objective,
    quality: dict[str, dict[str, Any]] | None = None,
    pool_size: int | None = None,
) -> str:
    """The card the user approves before anything runs."""
    lines = ["Objective", f"  Filters:  {describe_filters(objective.filters)}"]
    if pool_size is not None:
        lines[-1] += f"  ({pool_size:,} players)"
    for index, condition in enumerate(objective.conditions):
        prefix = "  Must:     " if index == 0 else "            "
        lines.append(prefix + condition_text(condition))
    lines.append(f"  Rank by:  {objective.rank_by.describe()}")
    lines.append(f"  Show:     {objective.count}")
    for phrase, meaning in objective.readings:
        lines.append(f'  Reading:  "{phrase}" = {meaning}')
    for target, report in (quality or {}).items():
        warning = report.get("verdict") in ("weak", "not predictable")
        lines.append(("  ⚠ " if warning else "  Model:    ") + quality_line(target, report))
    return "\n".join(lines)


def condition_text(condition: Condition) -> str:
    if condition.target == "price_vs_peers":
        return condition.describe()
    return f"{condition.describe()}, at least {condition.min_chance:.0%} likely"


def target_line(
    target: str,
    values: dict[str, float],
    fair: tuple[float, float, float] | None,
    scale: float,
) -> str:
    info = get_target(target)
    estimate, low, high = values["estimate"], values["low"], values["high"]
    if target == "price_vs_fair_value":
        text = f"Price ≈ {estimate:.2f}× fair value"
        if fair:
            text += (
                f" (fair value ≈ {_amount(fair[1] * scale)}, likely "
                f"{_amount(fair[0] * scale)}–{_amount(fair[2] * scale).removeprefix('€')})"
            )
        return text
    if target == "price_vs_peers":
        return f"Price ≈ {estimate:.2f}× the median of similar players"
    digits = 0 if info.scale[1] > 20 else 1
    return (
        f"{info.label.capitalize()} ≈ {estimate:.{digits}f} "
        f"(likely {low:.{digits}f}–{high:.{digits}f})"
    )


def result_view(
    result: ObjectiveResult,
    store: VisibleStore,
    *,
    explanations: dict[int, str] | None = None,
    note: str = "",
    fair_values: dict[int, tuple[float, float, float]] | None = None,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Everything a result shows, as data (the terminal and the web page both draw from this)."""
    objective = result.objective
    ids = [row["player_id"] for row in result.shortlist]
    details = {
        row["player_id"]: row
        for row in store.get_players(ids, currency_scale=scale, with_estimates=True)
    }
    stats = store.season_stats(ids)
    players = []
    for rank, row in enumerate(result.shortlist, 1):
        player = details[row["player_id"]]
        lines, shown = [], set()
        for index, condition in enumerate(objective.conditions):
            values = row["targets"].get(condition.target)
            chance = row["chances"][index]
            if values is None or condition.target in shown:
                continue
            text = target_line(
                condition.target, values, (fair_values or {}).get(row["player_id"]), scale
            )
            if condition.target != "price_vs_peers" and chance is not None:
                text += f" · {_chance(chance)} chance of {condition.describe()}"
            lines.append({"target": condition.target, "text": text, "chance": chance})
            shown.add(condition.target)
        ranking = objective.rank_by
        if ranking.target not in shown and ranking.target in row["targets"]:
            text = target_line(
                ranking.target,
                row["targets"][ranking.target],
                (fair_values or {}).get(row["player_id"]),
                scale,
            )
            if "rank_chance" in row:
                text += f" · {_chance(row['rank_chance'])} chance"
            lines.append({"target": ranking.target, "text": text, "chance": row.get("rank_chance")})
        players.append(
            {
                "rank": rank,
                "player_id": row["player_id"],
                "name": player["name"],
                "age": player["age"],
                "club": player["club"] or "no club",
                "price": price(player),
                "positions": player["natural_positions"] + player["accomplished_positions"],
                "lines": lines,
                "targets": row["targets"],
                "explanation": (explanations or {}).get(row["player_id"]),
                "season": format_season_stats(
                    stats.get(row["player_id"]), "GK" in player["natural_positions"]
                ),
            }
        )
    return {
        "funnel": [
            {"count": stage.remaining, "label": stage.label, "unjudged": stage.no_data}
            for stage in result.funnel
        ],
        "players": players,
        "note": note.strip(),
        "readings": [{"phrase": p, "meaning": m} for p, m in objective.readings],
        "suggestions": result.suggestions,
        "quality": [
            {
                "target": target,
                "text": quality_line(target, report),
                "warning": report.get("verdict") in ("weak", "not predictable"),
            }
            for target, report in result.quality.items()
        ],
    }


def funnel_text(view: dict[str, Any]) -> str:
    return " → ".join(
        f"{stage['count']:,} {stage['label']}"
        + (f" ({stage['unjudged']:,} couldn't be judged)" if stage["unjudged"] else "")
        for stage in view["funnel"]
    )


def render_result(
    result: ObjectiveResult,
    store: VisibleStore,
    *,
    explanations: dict[int, str] | None = None,
    note: str = "",
    fair_values: dict[int, tuple[float, float, float]] | None = None,
    scale: float = 1.0,
) -> str:
    view = result_view(
        result, store, explanations=explanations, note=note, fair_values=fair_values, scale=scale
    )
    lines = [funnel_text(view), ""]
    for player in view["players"]:
        lines.append(
            f"{player['rank']}. {player['name']} · {player['age']} · {player['club']} · "
            f"{player['price']}"
        )
        lines.extend("   " + line["text"] for line in player["lines"])
        if player["explanation"]:
            lines.append("   " + player["explanation"])
        if player["season"]:
            lines.append("   " + player["season"])
    if not view["players"]:
        lines.append("No player meets every condition.")
    if view["note"]:
        lines.extend(["", view["note"]])
    for reading in view["readings"]:
        lines.append(f'"{reading["phrase"]}" was read as: {reading["meaning"]}.')
    if view["suggestions"]:
        lines.append("")
        lines.extend(f"Tip: {tip}" for tip in view["suggestions"])
    lines.append("")
    lines.extend(item["text"] + "." for item in view["quality"])
    return "\n".join(lines).rstrip()


def objective_view(
    objective: Objective,
    quality: dict[str, dict[str, Any]] | None = None,
    pool_size: int | None = None,
) -> dict[str, Any]:
    """The objective card as data, for the web page (render_objective is its text form)."""
    return {
        "filters": describe_filters(objective.filters),
        "pool": pool_size,
        "conditions": [condition_text(condition) for condition in objective.conditions],
        "rank_by": objective.rank_by.describe(),
        "rank_mode": objective.rank_by.mode,
        "count": objective.count,
        "readings": [{"phrase": p, "meaning": m} for p, m in objective.readings],
        "quality": [
            {
                "target": target,
                "text": quality_line(target, report),
                "warning": report.get("verdict") in ("weak", "not predictable"),
            }
            for target, report in (quality or {}).items()
        ],
        "objective": objective.to_dict(),
    }
