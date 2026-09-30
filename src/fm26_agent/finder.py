"""Scouting without an LLM: explicit filters in, a checked TabPFN shortlist out.

`fm26-agent find --position MC --age-max 20 --max-value 20M` needs only a TabPFN key. It runs
the same tools and the same answer checks as the chat agent; the short explanations are written
by code from each player's standout attributes.
"""

from __future__ import annotations

import json
from typing import Any

from .agent import AgentResult, ScoutingAgent
from .tools import PREDICTION_TOOL, ScoutingTools
from .visible_db import VisibleStore, profile_attributes


class AmbiguousPlayerError(ValueError):
    """--like matched several players; the message lists them with ids."""


def resolve_player(store: VisibleStore, name_or_id: str) -> dict[str, Any]:
    """The one player --like refers to: a player_id, a unique name, or a unique partial name."""
    text = name_or_id.strip()
    if text.isdigit():
        found = store.get_players([int(text)])
        if found:
            return found[0]
    matches = store.lookup(text)
    if not matches:
        raise ValueError(f"No player called {text!r} in this save")
    if len(matches) > 1:
        listing = "\n".join(
            f"  {row['player_id']}  {row['name']} · {row['age']} · {row['club'] or 'no club'}"
            for row in matches
        )
        raise AmbiguousPlayerError(
            f"Several players match {text!r}. Use the number instead, e.g. --like "
            f"{matches[0]['player_id']}:\n{listing}"
        )
    return matches[0]


def describe(player: dict[str, Any]) -> str:
    """One plain sentence from what is visible: positions and the three best attributes."""
    goalkeeper = "GK" in player["natural_positions"]
    names = profile_attributes(goalkeeper)
    best = sorted(
        (name for name in names if player.get(name) is not None),
        key=lambda name: (-player[name], name),
    )[:3]
    positions = "/".join(player["natural_positions"]) or "no natural position"
    strengths = ", ".join(f"{name.replace('_', ' ')} {player[name]:.0f}" for name in best)
    return f"{positions}. Best visible attributes: {strengths}."


def find_players(
    store: VisibleStore,
    tools: ScoutingTools,
    *,
    count: int = 5,
    rank_by: str = "expected",
    like: str | None = None,
    **filters: Any,
) -> AgentResult:
    """Search, score the whole pool with TabPFN, and return the checked top `count`."""
    constraints = {key: value for key, value in filters.items() if value not in (None, [], "")}
    target = None
    if like:
        target = resolve_player(store, like)
        constraints["similar_to"] = target["player_id"]
    query = "find " + json.dumps(constraints | {"count": count, "rank_by": rank_by})
    result = AgentResult(query=query)
    search = tools.call("search_players", constraints)
    ranked = tools.call(
        PREDICTION_TOOL, {"search_id": search["search_id"], "top_k": count, "rank_by": rank_by}
    )
    ids = ranked["player_ids"]
    details = {row["player_id"]: row for row in store.get_players(ids)}
    answer = {
        "constraints": constraints,
        "requested_count": count,
        "ranking": rank_by,
        "recommendations": [
            {"player_id": player_id, "explanation": describe(details[player_id])}
            for player_id in ids
        ],
        "note": (
            f"Players whose visible profile looks most like {target['name']} "
            f"({target['age']}, {target['club'] or 'no club'}), ranked by potential."
            if target
            else ""
        )
        + ("" if ids else " Nothing matched these filters."),
    }
    checker = ScoutingAgent(backend=None, tools=tools)  # no LLM: only its answer checks
    checker._constraint_lock = None
    checker._finalize(json.dumps(answer), result)
    result.prediction_operations = tools.prediction_operations
    result.note = result.note.strip()
    return result
