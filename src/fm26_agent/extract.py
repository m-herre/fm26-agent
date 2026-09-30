from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .schema import POSITION_CODES, VISIBLE_ATTRIBUTES


@dataclass
class ExtractedPlayer:
    visible: dict[str, Any]
    potential_ability: int | None


@dataclass
class ExtractionResult:
    players: list[ExtractedPlayer]
    save_date: date
    game: str
    build: str
    warnings: list[str]


def _enum_label(value: Any) -> str:
    label = getattr(value, "label", value)
    name = getattr(label, "name", None)
    return str(name if name is not None else label)


def _position_list(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        code = _enum_label(value).upper()
        if code in POSITION_CODES and code not in result:
            result.append(code)
    return result


def _preferred_foot(player: Any) -> str:
    left = int(getattr(player, "left_foot", 0) or 0)
    right = int(getattr(player, "right_foot", 0) or 0)
    if max(left, right) == 0:
        return "unknown"
    if abs(left - right) <= 2 and min(left, right) >= 10:
        return "both"
    return "left" if left > right else "right"


def _save_date(info: Any) -> date:
    for attr in ("date", "game_date", "current_date", "in_game_date"):
        value = getattr(info, attr, None)
        if isinstance(value, date):
            return value
    raise ValueError("fmsave did not expose a usable in-game date")


def record_to_player(player: Any, save_date: date, eur_rate: float) -> ExtractedPlayer:
    attrs = player.attributes
    contract = getattr(player, "contract", None)
    contract_end = getattr(contract, "end", None) if contract else None
    contract_days = (contract_end - save_date).days if isinstance(contract_end, date) else None
    transfer_value = getattr(player, "transfer_value", None)
    wage = getattr(contract, "wage", None) if contract else None
    traits = sorted(
        {
            _enum_label(value)
            for value in (getattr(player, "traits", ()) or ())
            if _enum_label(value) != "UNKNOWN"
        }
    )
    visible: dict[str, Any] = {
        "player_id": int(player.uid),
        "name": getattr(player, "name", None) or getattr(player, "full_name", None) or "Unknown",
        "age": getattr(player, "age", None),
        "club": getattr(player, "club_name", None),
        "club_uid": getattr(player, "club_uid", None),
        "nation_id": getattr(player, "nation_id", None),
        "height_cm": getattr(player, "height_cm", None),
        "value_eur": float(transfer_value) * eur_rate if transfer_value is not None else None,
        "wage_eur": float(wage) * eur_rate if wage is not None else None,
        "contract_end": contract_end.isoformat() if isinstance(contract_end, date) else None,
        "contract_days_remaining": contract_days,
        "on_loan": bool(getattr(player, "on_loan", False))
        if getattr(player, "on_loan", None) is not None
        else None,
        "natural_positions": _position_list(getattr(player, "natural_positions", ())),
        "accomplished_positions": _position_list(getattr(player, "accomplished_positions", ())),
        "preferred_foot": _preferred_foot(player),
        "traits": traits,
        "split": "unassigned",
    }
    for attribute in VISIBLE_ATTRIBUTES:
        visible[attribute] = getattr(attrs, attribute, None)
    ability = getattr(player, "ability", None)
    potential = getattr(ability, "potential", None) if ability else None
    if potential is not None and not 1 <= int(potential) <= 200:
        potential = None
    return ExtractedPlayer(
        visible=visible, potential_ability=int(potential) if potential is not None else None
    )


def read_save(
    path: str | Path, eur_rate: float, allow_reader_warnings: bool = False
) -> ExtractionResult:
    try:
        import fmsave
    except ImportError as exc:
        raise RuntimeError(
            "fmsave is not installed; install this project with Python 3.12"
        ) from exc
    save_path = Path(path).expanduser().resolve()
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fmsave.open(save_path, strict=False) as career:
            info = career.info
            game_date = _save_date(info)
            table = career.players()
            players = [record_to_player(player, game_date, eur_rate) for player in table]
        captured = [str(item.message) for item in caught]
    reader_warning_type = getattr(fmsave, "ReaderCheckWarning", Warning)
    reader_warnings = [
        str(item.message) for item in caught if issubclass(item.category, reader_warning_type)
    ]
    if reader_warnings and not allow_reader_warnings:
        joined = "\n".join(reader_warnings)
        raise RuntimeError(
            f"fmsave player reader checks failed:\n{joined}\nRerun with --allow-reader-warnings after review."
        )
    if not players:
        raise ValueError("The save's player reader returned no players")
    return ExtractionResult(
        players=players,
        save_date=game_date,
        game=str(getattr(info, "game", "FM26")),
        build=str(getattr(info, "build", "unknown")),
        warnings=captured,
    )
