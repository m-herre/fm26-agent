"""The objective a user and the planner agree on, and its deterministic execution.

An objective is filters (what the save can filter by), conditions on hidden targets that each
need a minimum chance ("potential 160+, at least 25% likely"), one ranking and a count. Code runs
it: search the complete pool, let TabPFN predict every target it needs, apply the conditions in
order (recording how many players each one removes), rank the survivors and take the top. The LLM
never picks players; it only helps write the objective and explain the result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from .custom_tasks import TaskLab, chance_of
from .prediction import HIGH_LEVEL, LOW_LEVEL, MEDIAN
from .schema import normalize_positions
from .targets import TARGETS, define_formula_target, formula_definition, get_target
from .tools import RANKINGS, SEARCH_PROPERTIES
from .visible_db import VisibleStore

MAX_CONDITIONS = 4
PRICE_WORDS = {
    "price_vs_fair_value": "of his fair value",
    "price_vs_peers": "of what similar players cost",
}
FILTER_PROPERTIES = {
    key: value for key, value in SEARCH_PROPERTIES.items() if key not in ("limit", "offset")
}
LEVEL = {"type": "number"}
OBJECTIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["filters", "conditions", "rank_by", "count"],
    "properties": {
        "filters": {
            "type": "object",
            "properties": FILTER_PROPERTIES,
            "additionalProperties": False,
        },
        "conditions": {
            "type": "array",
            "maxItems": MAX_CONDITIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "min_chance"],
                "properties": {
                    "target": {"type": "string"},
                    "at_least": LEVEL,
                    "at_most": LEVEL,
                    "min_chance": {"type": "number", "minimum": 0.05, "maximum": 0.95},
                },
                "oneOf": [{"required": ["at_least"]}, {"required": ["at_most"]}],
            },
        },
        "rank_by": {
            "type": "object",
            "additionalProperties": False,
            "required": ["target"],
            "properties": {
                "target": {"type": "string"},
                "mode": {"type": "string", "enum": list(RANKINGS), "default": "expected"},
                "at_least": LEVEL,
                "at_most": LEVEL,
            },
        },
        "count": {"type": "integer", "minimum": 1, "maximum": 25},
        "custom_targets": {
            "type": "array",
            "maxItems": 3,
            "description": "Definitions of agent-defined targets the objective uses (added by the application).",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "combine"],
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string", "maxLength": 40},
                    "combine": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
            },
        },
        "readings": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["phrase", "meaning"],
                "properties": {
                    "phrase": {"type": "string", "maxLength": 80},
                    "meaning": {"type": "string", "maxLength": 200},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class Condition:
    target: str
    side: str  # at_least | at_most
    level: float
    min_chance: float

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, self.side: self.level, "min_chance": self.min_chance}

    def chance(self, curve: np.ndarray) -> float:
        return chance_of(curve, self.level, self.side, get_target(self.target).whole)

    def describe(self) -> str:
        """e.g. 'potential 160 or higher' / 'costing at most 80% of his fair value'."""
        target = get_target(self.target)
        if self.target in PRICE_WORDS:
            side = "at most" if self.side == "at_most" else "at least"
            return f"costing {side} {self.level:.0%} {PRICE_WORDS[self.target]}"
        return (
            f"{target.label} {self.level:g} {'or lower' if self.side == 'at_most' else 'or higher'}"
        )


@dataclass(frozen=True)
class Ranking:
    target: str
    mode: str = "expected"
    side: str | None = None
    level: float | None = None

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {"target": self.target, "mode": self.mode}
        if self.side:
            item[self.side] = self.level
        return item

    def describe(self) -> str:
        target = get_target(self.target)
        if self.mode == "chance":
            return f"chance of {Condition(self.target, self.side or 'at_least', self.level or 0, 0.5).describe()}"
        best = "lowest" if target.better == "low" else "highest"
        return {
            "expected": f"{best} {target.label}",
            "ceiling": f"best case {target.label}",
            "safe": f"safest {target.label} (worst case)",
        }[self.mode]


@dataclass(frozen=True)
class Objective:
    filters: dict[str, Any]
    conditions: tuple[Condition, ...]
    rank_by: Ranking
    count: int = 5
    readings: tuple[tuple[str, str], ...] = ()

    @property
    def targets(self) -> list[str]:
        """Every target the objective needs, conditions first, without repeats."""
        return list(dict.fromkeys([c.target for c in self.conditions] + [self.rank_by.target]))

    def to_dict(self) -> dict[str, Any]:
        custom = [
            formula_definition(get_target(name))
            for name in self.targets
            if get_target(name).formula
        ]
        return {
            "filters": self.filters,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "rank_by": self.rank_by.to_dict(),
            "count": self.count,
            "readings": [{"phrase": p, "meaning": m} for p, m in self.readings],
        } | ({"custom_targets": custom} if custom else {})

    @classmethod
    def from_dict(cls, data: Any, available: Sequence[str] | None = None) -> Objective:
        """Validate a proposed objective (schema, then meaning) and build it."""
        errors = sorted(
            Draft202012Validator(OBJECTIVE_SCHEMA).iter_errors(data), key=lambda e: list(e.path)
        )
        if errors:
            where = ".".join(str(part) for part in errors[0].path) or "objective"
            raise ValueError(f"Invalid objective at {where}: {errors[0].message}")
        filters = {key: value for key, value in data["filters"].items() if value is not None}
        for key, value in filters.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"filters.{key} must be finite")
        if filters.get("position") is not None:
            filters["position"] = normalize_positions(filters["position"])
        for lower, upper in (
            ("age_min", "age_max"),
            ("value_min_eur", "value_max_eur"),
            ("height_min_cm", "height_max_cm"),
        ):
            if filters.get(lower) is not None and filters.get(upper) is not None:
                if filters[lower] > filters[upper]:
                    raise ValueError(f"filters.{lower} cannot exceed {upper}")
        allowed = set(available) if available is not None else set(TARGETS)
        for item in data.get("custom_targets", []):
            target = define_formula_target(item["name"], item.get("label", ""), item["combine"])
            if all(part in allowed for part, _ in target.formula or ()):
                allowed.add(target.name)

        def target_of(name: str) -> str:
            if name not in TARGETS:
                raise ValueError(
                    f"Unknown target {name!r}; use one of {', '.join(sorted(allowed))}"
                )
            if name not in allowed:
                raise ValueError(f"This save cannot predict {name}")
            return name

        def level_of(name: str, item: dict[str, Any]) -> tuple[str | None, float | None]:
            side = "at_least" if "at_least" in item else "at_most" if "at_most" in item else None
            if side is None:
                return None, None
            level = float(item[side])
            low, high = get_target(name).scale
            if not (math.isfinite(level) and low <= level <= high):
                raise ValueError(f"{name} level must be between {low:g} and {high:g}")
            return side, level

        conditions = []
        for item in data["conditions"]:
            name = target_of(item["target"])
            side, level = level_of(name, item)
            conditions.append(Condition(name, side, level, float(item["min_chance"])))
        rank = data["rank_by"]
        name = target_of(rank["target"])
        side, level = level_of(name, rank)
        mode = rank.get("mode", "expected")
        if mode == "chance" and side is None:
            raise ValueError("rank_by mode chance needs at_least or at_most")
        return cls(
            filters,
            tuple(conditions),
            Ranking(name, mode, side, level),
            int(data["count"]),
            tuple((r["phrase"], r["meaning"]) for r in data.get("readings", [])),
        )

    @classmethod
    def loads(cls, text: str, available: Sequence[str] | None = None) -> Objective:
        return cls.from_dict(json.loads(text), available)


# ---------------------------------------------------------------------------- execution


@dataclass
class Stage:
    label: str
    remaining: int
    no_data: int = 0  # players the condition couldn't judge (e.g. no stored price)


@dataclass
class ObjectiveResult:
    objective: Objective
    pool_size: int
    funnel: list[Stage]
    shortlist: list[dict[str, Any]]  # player_id, per-target estimate/low/high, per-condition chance
    quality: dict[str, dict[str, Any]]
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective.to_dict(),
            "pool_size": self.pool_size,
            "funnel": [asdict(stage) for stage in self.funnel],
            "shortlist": self.shortlist,
            "quality": self.quality,
            "suggestions": self.suggestions,
        }


def rank_value(curve: np.ndarray, ranking: Ranking) -> tuple[float, float]:
    """(primary, tie-break) — both higher is better. Tie-break is the signed expected value."""
    target = get_target(ranking.target)
    sign = -1.0 if target.better == "low" else 1.0
    expected = sign * float(curve[MEDIAN])
    if ranking.mode == "chance":
        assert ranking.side is not None and ranking.level is not None
        return chance_of(curve, ranking.level, ranking.side, target.whole), expected
    good_end, bad_end = (HIGH_LEVEL, LOW_LEVEL) if sign > 0 else (LOW_LEVEL, HIGH_LEVEL)
    if ranking.mode == "ceiling":
        return sign * float(curve[good_end]), expected
    if ranking.mode == "safe":
        return sign * float(curve[bad_end]), expected
    return expected, expected


def execute(
    objective: Objective,
    store: VisibleStore,
    lab: TaskLab,
    *,
    include_unknown_value: bool = True,
    currency_scale: float = 1.0,
) -> ObjectiveResult:
    """Run an objective. Deterministic given the saved fits: same objective, same shortlist."""
    ids = store.matching_ids(
        **objective.filters,
        include_unknown_value=include_unknown_value,
        currency_scale=currency_scale,
    )
    players = store.get_players(ids)
    curves = {target: lab.distribution(target, players) for target in objective.targets}
    quality = {target: lab.build_quality(target) for target in objective.targets}
    chances = {
        index: {
            player_id: condition.chance(curve)
            for player_id, curve in curves[condition.target].items()
        }
        for index, condition in enumerate(objective.conditions)
    }
    funnel = [Stage("match the filters", len(ids))]
    survivors = list(ids)
    for index, condition in enumerate(objective.conditions):
        judged = chances[index]
        kept = [pid for pid in survivors if judged.get(pid, -1.0) >= condition.min_chance]
        no_data = sum(pid not in judged for pid in survivors)
        funnel.append(
            Stage(
                f"{condition.describe()} (at least {condition.min_chance:.0%} likely)",
                len(kept),
                no_data,
            )
        )
        survivors = kept
    ranking = objective.rank_by
    ranked_curves = curves[ranking.target]
    unrankable = sum(pid not in ranked_curves for pid in survivors)
    survivors = [pid for pid in survivors if pid in ranked_curves]
    if unrankable:
        funnel.append(
            Stage(f"have a {get_target(ranking.target).label}", len(survivors), unrankable)
        )
    survivors.sort(key=lambda pid: (*(-v for v in rank_value(ranked_curves[pid], ranking)), pid))
    top = survivors[: objective.count]
    shortlist = []
    for pid in top:
        row: dict[str, Any] = {"player_id": pid, "targets": {}, "chances": []}
        for target in objective.targets:
            curve = curves[target].get(pid)
            if curve is not None:
                row["targets"][target] = {
                    "estimate": round(float(curve[MEDIAN]), 3),
                    "low": round(float(curve[LOW_LEVEL]), 3),
                    "high": round(float(curve[HIGH_LEVEL]), 3),
                }
        row["chances"] = [chances[index].get(pid) for index in range(len(objective.conditions))]
        if ranking.mode == "chance":
            row["rank_chance"] = rank_value(ranked_curves[pid], ranking)[0]
        shortlist.append(row)
    result = ObjectiveResult(objective, len(ids), funnel, shortlist, quality)
    result.suggestions = suggestions(objective, ids, chances, len(top))
    return result


def suggestions(
    objective: Objective,
    ids: Sequence[int],
    chances: dict[int, dict[int, float]],
    found: int,
) -> list[str]:
    """When fewer players survive than asked for: what each condition costs, and the exact
    chance threshold that would give enough players (from predictions already made)."""
    if found >= objective.count:
        return []
    if len(ids) < objective.count:
        return [f"Only {len(ids)} players match the filters themselves."]
    tips = []
    for index, condition in enumerate(objective.conditions):
        others = [
            pid
            for pid in ids
            if all(
                chances[other].get(pid, -1.0) >= objective.conditions[other].min_chance
                for other in range(len(objective.conditions))
                if other != index
            )
        ]
        judged = sorted(
            (chances[index][pid] for pid in others if pid in chances[index]), reverse=True
        )
        label = condition.describe()
        if len(judged) >= objective.count:
            needed = judged[objective.count - 1]
            if needed < condition.min_chance and needed >= 0.05:
                tips.append(
                    f"Accepting a {needed:.0%} chance for {label} (instead of "
                    f"{condition.min_chance:.0%}) gives {objective.count} players."
                )
                continue
        tips.append(f"Without the condition on {label}: {len(others)} players.")
    return tips
