"""Checks that need a human: compare a few players with the game to validate extracted data.

`spotcheck_sample` picks players to look up; `calibrate` turns the in-game values you read off
into the euro multiplier. Neither changes the databases or the models.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .private_db import PrivateStore
from .visible_db import VisibleStore

MAX_SPREAD = 0.05  # in-game values are rounded, so allow a few percent between players
MIN_STORED_VALUE = 2_000_000  # larger values lose less precision to the game's rounding
_AMOUNT = re.compile(r"^\s*[€£$]?\s*(\d[\d,]*(?:\.\d+)?|\.\d+)\s*([kKmMbB]?)\s*$")
_UNITS = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9}


def parse_amount(text: str) -> float:
    """Parse an in-game figure such as '€4.5M', '850K' or '1,250,000' into a number."""
    match = _AMOUNT.match(text.strip())
    if not match:
        raise ValueError(
            f"Cannot read an amount from {text!r}; use forms like 4.5M, 850K or 1250000"
        )
    return float(match.group(1).replace(",", "")) * _UNITS[match.group(2).lower()]


def spotcheck_sample(
    store: VisibleStore, private: PrivateStore, count: int = 8
) -> list[dict[str, Any]]:
    """Players with a stored value and a unique name, spread evenly across the PA range."""
    names: dict[str, int] = {}
    players = store.get_players(store.all_ids())
    for row in players:
        names[row["name"].casefold()] = names.get(row["name"].casefold(), 0) + 1
    labels = private.get([row["player_id"] for row in players])
    pool = [
        row
        for row in players
        if row["value_eur"] is not None
        and row["value_eur"] >= MIN_STORED_VALUE
        and names[row["name"].casefold()] == 1
        and labels.get(row["player_id"], {}).get("potential_ability") is not None
    ]
    pool.sort(key=lambda row: (labels[row["player_id"]]["potential_ability"], row["player_id"]))
    if len(pool) <= count:
        chosen = pool
    else:
        chosen = [pool[round(i * (len(pool) - 1) / (count - 1))] for i in range(count)]
    return [
        {
            "name": row["name"],
            "age": row["age"],
            "club": row["club"],
            "stored_value": row["value_eur"],
            "extracted_pa": labels[row["player_id"]]["potential_ability"],
        }
        for row in chosen
    ]


def calibrate(store: VisibleStore, observations: Sequence[tuple[str, float]]) -> dict[str, Any]:
    """Estimate euros per internal unit from players whose in-game value you looked up."""
    rows = []
    for name, euros in observations:
        matches = store.find_by_name(name)
        if len(matches) != 1:
            raise ValueError(
                f"{name!r} matches {len(matches)} players; use the exact, unique name from spotcheck"
            )
        stored = matches[0]["value_eur"]
        if stored is None:
            raise ValueError(f"{name!r} has no stored value in the save, so it cannot be used")
        if euros <= 0:
            raise ValueError(f"The in-game value for {name!r} must be positive")
        rows.append(
            {
                "name": matches[0]["name"],
                "stored": stored,
                "in_game": euros,
                "ratio": euros / stored,
            }
        )
    if not rows:
        raise ValueError("Give at least one player, for example: calibrate 'Name=4.5M'")
    ratios = [row["ratio"] for row in rows]
    median = statistics.median(ratios)
    spread = (max(ratios) - min(ratios)) / median
    return {
        "players": rows,
        "eur_per_internal_unit": round(median, 4),
        "spread": spread,
        "consistent": len(rows) >= 2 and spread <= MAX_SPREAD,
    }


def apply_to_config(config_path: Path, rate: float) -> None:
    """Write the multiplier into config.toml (creating it if needed) and mark it calibrated."""
    money = f"[money]\neur_per_internal_unit = {rate}\ncalibrated = true\n"
    if not config_path.exists():
        config_path.write_text(money, encoding="utf-8")
        return
    text = config_path.read_text(encoding="utf-8")
    if not re.search(r"^\[money\]", text, flags=re.M):
        config_path.write_text(
            text.rstrip("\n") + ("\n\n" if text.strip() else "") + money, encoding="utf-8"
        )
        return
    text, rate_hits = re.subn(
        r"^eur_per_internal_unit\s*=.*$", f"eur_per_internal_unit = {rate}", text, flags=re.M
    )
    text, flag_hits = re.subn(r"^calibrated\s*=.*$", "calibrated = true", text, flags=re.M)
    if not (rate_hits == flag_hits == 1):
        raise ValueError(
            "config.toml needs one eur_per_internal_unit and one calibrated line under [money]"
        )
    config_path.write_text(text, encoding="utf-8")
