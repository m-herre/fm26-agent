"""The glossary of hidden things the agent may build a prediction task around.

Each entry is something the save knows but a scout cannot see. The agent chooses one by name; it
never sees the values, only this description of what the target means, its scale and which end
is good. Features are always the fixed visible schema, and every name here is also a forbidden
feature (schema.assert_safe_features), so a target can never leak into its own inputs.
"""

from __future__ import annotations

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
    )
}

# What the targets table stores. Potential has its own labels table; derived targets are
# computed when read.
STORED_TARGETS = tuple(
    name for name, target in TARGETS.items() if not target.derived and name != "potential_ability"
)
PERSONALITY = tuple(name for name, target in TARGETS.items() if target.group == "personality")
HIDDEN_ATTRIBUTE_TARGETS = tuple(
    name for name, target in TARGETS.items() if target.group == "hidden attribute"
)


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
