"""The glossary of hidden things the agent may build a prediction task around.

Each entry is something the save knows but a scout cannot see. The agent chooses one by name; it
never sees the values, only this description of what the target means, its scale and which end
is good. Features are always the fixed visible schema, and every name here is also a forbidden
feature (schema.assert_safe_features), so a target can never leak into its own inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TARGETS_VERSION = 1  # bump when the stored targets change


@dataclass(frozen=True)
class Target:
    name: str
    label: str  # plain words for players, e.g. "current ability"
    description: str  # what it means in FM and which requests it answers
    scale: tuple[float, float]
    better: str  # "high" or "low"
    group: str
    derived: bool = False  # computed from other stored targets, not read from the save
    whole: bool = True  # whole-number scale: "15 or better" counts from 14.5
    # Agent-defined targets: a weighted average of stored 1-20 targets, e.g. "mentality".
    formula: tuple[tuple[str, float], ...] | None = None


def _hidden(name: str, label: str, description: str, better: str = "high") -> Target:
    return Target(name, label, description, (1.0, 20.0), better, "hidden attribute")


def _personality(name: str, label: str, description: str, better: str = "high") -> Target:
    return Target(name, label, description, (1.0, 20.0), better, "personality")


TARGETS: dict[str, Target] = {
    target.name: target
    for target in (
        Target(
            "potential_ability",
            "potential",
            "How good he can become (hidden PA). For prospects and wonderkids; the saved potential "
            "model already covers it, so prefer predict_player_potential.",
            (1.0, 200.0),
            "high",
            "ability",
        ),
        Target(
            "current_ability",
            "current ability",
            "How good he is right now (hidden CA). For 'in his prime', 'ready now', 'proven', "
            "'best right now', 'can start straight away'.",
            (1.0, 200.0),
            "high",
            "ability",
        ),
        Target(
            "growth_room",
            "room to grow",
            "Potential minus current ability: how much better he can still get. For 'still "
            "improving', 'late bloomer', 'not the finished article'.",
            (0.0, 200.0),
            "high",
            "ability",
            derived=True,
        ),
        _hidden(
            "consistency",
            "consistency",
            "How often he plays to his level. For 'reliable every week', 'consistent'.",
        ),
        _hidden(
            "important_matches",
            "big-match temperament",
            "How he performs in finals and derbies. For 'big-game player', 'clutch'.",
        ),
        _hidden(
            "injury_proneness",
            "injury proneness",
            "How often he gets injured; LOW is good. For 'stays fit', 'not injury-prone'.",
            "low",
        ),
        _hidden(
            "dirtiness",
            "dirtiness",
            "How often he commits deliberate fouls; LOW is good. For 'clean', 'disciplined'.",
            "low",
        ),
        _hidden(
            "versatility",
            "versatility",
            "How well he adapts to positions he isn't trained for. For 'versatile', 'utility'.",
        ),
        _personality(
            "professionalism",
            "professionalism",
            "How hard and properly he trains and lives. For 'model professional', 'good attitude'.",
        ),
        _personality(
            "ambition",
            "ambition",
            "How much he wants to reach the top. For 'hungry', 'ambitious'.",
        ),
        _personality(
            "adaptability",
            "adaptability",
            "How quickly he settles in a new country. For 'will settle abroad'.",
        ),
        _personality(
            "loyalty",
            "loyalty",
            "How attached he gets to his club. For 'loyal', 'one-club man', 'won't ask to leave'.",
        ),
        _personality(
            "pressure",
            "handling pressure",
            "How well he copes with pressure and expectation. For 'handles pressure'.",
        ),
        _personality(
            "temperament",
            "temperament",
            "How calm he stays when provoked. For 'level-headed', 'won't get sent off'.",
        ),
        _personality(
            "sportsmanship",
            "sportsmanship",
            "How fairly he plays. For 'fair player', 'good sport'.",
        ),
        _personality(
            "controversy",
            "controversy",
            "How likely he is to cause trouble in the media; LOW is good. For 'no drama', "
            "'keeps quiet'.",
            "low",
        ),
        Target(
            "price_vs_fair_value",
            "price vs fair value",
            "His price divided by his fair value: what TabPFN thinks his visible profile is worth, "
            "learned from other players' prices (never his own). 0.7 means he costs 70% of what "
            "he's worth; LOW is good. For 'undervalued', 'bargain', 'cheap for what he is'. Only "
            "players with a stored price have one.",
            (0.0, 20.0),
            "low",
            "value",
            derived=True,
            whole=False,
        ),
        Target(
            "price_vs_peers",
            "price vs similar players",
            "His price divided by the median price of players at his position with similar "
            "(estimated) current ability. 0.7 means 30% cheaper than comparable players; LOW is "
            "good. A comparison, not a prediction. For 'cheaper than players like him'.",
            (0.0, 20.0),
            "low",
            "value",
            derived=True,
            whole=False,
        ),
    )
}
VALUE_TARGETS = tuple(name for name, target in TARGETS.items() if target.group == "value")

# What the targets table stores. Potential has its own labels table; derived targets are
# computed when read.
STORED_TARGETS = tuple(
    name for name, target in TARGETS.items() if not target.derived and name != "potential_ability"
)
PERSONALITY = tuple(name for name, target in TARGETS.items() if target.group == "personality")
HIDDEN_ATTRIBUTE_TARGETS = tuple(
    name for name, target in TARGETS.items() if target.group == "hidden attribute"
)


FORMULA_GROUPS = ("hidden attribute", "personality")
BUILT_IN = frozenset(TARGETS)
NAME = re.compile(r"^[a-z][a-z0-9_]{2,30}$")


def define_formula_target(
    name: str, label: str, combine: dict[str, float], description: str = ""
) -> Target:
    """Register a target the agent designed: a weighted average of stored 1-20 hidden values.

    Where low is good (injury proneness, dirtiness, controversy) the value is flipped (21 - x)
    first, so the result is always "high is good" on 1-20. Re-defining a name replaces it.
    """
    from .schema import VISIBLE_ATTRIBUTES

    if not NAME.match(name) or name in BUILT_IN or name in VISIBLE_ATTRIBUTES:
        raise ValueError(
            f"Custom target name {name!r} must be new, lowercase with underscores (3-31 chars)"
        )
    if not 2 <= len(combine) <= 6:
        raise ValueError("A custom target combines 2 to 6 hidden values")
    parts = []
    for part, weight in combine.items():
        target = TARGETS.get(part)
        if target is None or target.group not in FORMULA_GROUPS:
            raise ValueError(
                f"{part!r} can't be combined; use hidden attributes or personality: "
                + ", ".join(n for n, t in TARGETS.items() if t.group in FORMULA_GROUPS)
            )
        if not 0 < float(weight) <= 5:
            raise ValueError("Weights must be between 0 and 5")
        parts.append((part, float(weight)))
    words = ", ".join(
        f"{TARGETS[part].label}{' (reversed)' if TARGETS[part].better == 'low' else ''}"
        + (f" ×{weight:g}" if weight != 1 else "")
        for part, weight in parts
    )
    target = Target(
        name,
        _sentence_case(label.strip()[:40]) or name.replace("_", " "),
        (description.strip()[:200] + " " if description.strip() else "")
        + f"Agent-defined: average of {words}.",
        (1.0, 20.0),
        "high",
        "custom",
        derived=True,
        whole=False,
        formula=tuple(parts),
    )
    TARGETS[name] = target
    return target


def _sentence_case(label: str) -> str:
    """Labels sit mid-sentence ("88% chance of strong mentality 14 or higher"): lower the first
    letter unless the word is an acronym (e.g. "CA")."""
    if not label or label[:2].isupper():
        return label
    return label[0].lower() + label[1:]


def formula_definition(target: Target) -> dict[str, Any]:
    return {
        "name": target.name,
        "label": target.label,
        "combine": dict(target.formula or ()),
    }


def combine_values(target: Target, components: dict[str, dict[int, float]]) -> dict[int, float]:
    """The formula target's value for every player who has all its parts."""
    assert target.formula is not None
    ids = set.intersection(*(set(components[part]) for part, _ in target.formula))
    total = sum(weight for _, weight in target.formula)
    result = {}
    for player_id in ids:
        value = 0.0
        for part, weight in target.formula:
            raw = components[part][player_id]
            value += weight * (21 - raw if TARGETS[part].better == "low" else raw)
        result[player_id] = value / total
    return result


def forget_formula_targets() -> None:
    """Drop every agent-defined target (tests, or a fresh session)."""
    for name in [name for name in TARGETS if name not in BUILT_IN]:
        del TARGETS[name]


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise ValueError(f"Unknown target {name!r}; use one of {', '.join(TARGETS)}")
    return TARGETS[name]


def clean_value(name: str, value: Any) -> float | None:
    """A stored target value, or None when it is missing or outside the target's scale."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    low, high = TARGETS[name].scale
    return number if low <= number <= high else None


def glossary(names: list[str] | None = None) -> str:
    """One line per target for the agent's tool description: meaning, scale and good end."""
    lines = []
    for name in names or list(TARGETS):
        target = TARGETS[name]
        low, high = target.scale
        lines.append(f"{name} ({low:g}-{high:g}, {target.better} is good): {target.description}")
    return "\n".join(lines)
