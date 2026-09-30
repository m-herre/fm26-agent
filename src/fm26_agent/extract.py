from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .schema import POSITION_CODES, VISIBLE_ATTRIBUTES


class UnsupportedSaveError(ValueError):
    """The file is not a save this tool can read reliably; the message tells the user why."""


def supported_builds() -> tuple[str, ...]:
    """Game builds fmsave has layout tables for, or () if that cannot be determined."""
    try:
        from fmsave._layouts import known_builds

        return tuple(sorted(known_builds()))
    except Exception:
        return ()


def supported_saves_message() -> str:
    builds = supported_builds()
    which = ", ".join(builds) if builds else "the final FM26 update"
    return (
        f"Only Football Manager 26 saves from build {which} are supported. Saves from other "
        "Football Manager versions (FM25 and earlier) cannot be read. If your save comes from an "
        "older FM26 update, try loading it in the latest FM26 and saving it again."
    )


@dataclass
class SaveInspection:
    game: str
    build: str
    save_date: date
    supported: bool
    warnings: list[str]


SEASON_STATS_VERSION = 1  # bump when the stored season-stat fields change

MAX_PA_BELOW_CURRENT = 0.01  # the game never lets potential fall below current ability


@dataclass
class ExtractedPlayer:
    visible: dict[str, Any]
    potential_ability: int | None
    # Sanity-check input only: current ability is compared in memory and never stored or modelled.
    potential_below_current: bool | None = None


@dataclass
class ExtractionResult:
    players: list[ExtractedPlayer]
    save_date: date
    game: str
    build: str
    warnings: list[str]
    pa_below_current_fraction: float | None = None
    # Display-only: never a model input. Empty when the save has no stats or they cannot be read.
    season_stats: dict[int, dict[str, Any]] = field(default_factory=dict)


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


def record_to_player(player: Any, save_date: date) -> ExtractedPlayer:
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
        "value_eur": float(transfer_value)
        if transfer_value is not None
        else None,  # internal units
        "wage_eur": float(wage) if wage is not None else None,  # internal units
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
    current = getattr(ability, "current", None) if ability else None
    return ExtractedPlayer(
        visible=visible,
        potential_ability=int(potential) if potential is not None else None,
        potential_below_current=(
            int(potential) < int(current) if potential is not None and current is not None else None
        ),
    )


def _season_stats(career: Any) -> dict[int, dict[str, Any]]:
    """This season's totals per player who has played, summed over the teams they played for."""
    totals: dict[int, dict[str, Any]] = {}
    rating_weight: dict[int, float] = {}
    for row in career.player_season_stats():
        if getattr(row.kind, "value", row.kind) != "overall" or not row.minutes:
            continue
        item = totals.setdefault(
            int(row.player_uid),
            {
                key: 0
                for key in (
                    "appearances",
                    "starts",
                    "minutes",
                    "goals",
                    "assists",
                    "player_of_the_match",
                    "clean_sheets",
                )
            }
            | {"expected_goals": 0.0, "expected_assists": 0.0, "average_rating": None},
        )
        item["starts"] += row.starts or 0
        item["appearances"] += (row.starts or 0) + (row.substitute_appearances or 0)
        item["minutes"] += row.minutes
        item["goals"] += row.goals or 0
        item["assists"] += row.assists or 0
        item["player_of_the_match"] += row.player_of_the_match or 0
        item["clean_sheets"] += row.clean_sheets or 0
        item["expected_goals"] += row.expected_goals or 0.0
        item["expected_assists"] += row.expected_assists or 0.0
        if row.average_rating and row.rated_appearances:
            weight = rating_weight.get(int(row.player_uid), 0.0)
            previous = item["average_rating"] or 0.0
            total = weight + row.rated_appearances
            item["average_rating"] = (
                previous * weight + row.average_rating * row.rated_appearances
            ) / total
            rating_weight[int(row.player_uid)] = total
    for item in totals.values():
        for key in ("expected_goals", "expected_assists"):
            item[key] = round(item[key], 1)
        if item["average_rating"] is not None:
            item["average_rating"] = round(item["average_rating"], 2)
    return totals


def _import_fmsave():
    try:
        import fmsave
    except ImportError as exc:
        raise RuntimeError(
            "fmsave is not installed; install this project with Python 3.12"
        ) from exc
    return fmsave


def _unsupported(fmsave: Any, exc: Exception, path: Path) -> UnsupportedSaveError:
    if isinstance(exc, fmsave.NotAFmSaveError):
        return UnsupportedSaveError(
            f"{path.name} is not a Football Manager save file. {supported_saves_message()}"
        )
    return UnsupportedSaveError(f"{exc}\n{supported_saves_message()}")


def _unknown_build_messages(fmsave: Any, caught: list[Any]) -> list[str]:
    return [
        str(item.message)
        for item in caught
        if issubclass(item.category, fmsave.UnknownBuildWarning)
    ]


def inspect_save(path: str | Path) -> SaveInspection:
    """Check which game version a save comes from without reading its player table."""
    fmsave = _import_fmsave()
    save_path = Path(path).expanduser().resolve()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with fmsave.open(save_path, strict=False) as career:
                info = career.info
                game_date = _save_date(info)
        except (fmsave.UnsupportedGameError, fmsave.NotAFmSaveError) as exc:
            raise _unsupported(fmsave, exc, save_path) from exc
    unknown_build = _unknown_build_messages(fmsave, caught)
    return SaveInspection(
        game=str(getattr(info, "game", "FM26")),
        build=str(getattr(info, "build", "unknown")),
        save_date=game_date,
        supported=not unknown_build,
        warnings=unknown_build,
    )


def read_save(path: str | Path, allow_reader_warnings: bool = False) -> ExtractionResult:
    fmsave = _import_fmsave()
    save_path = Path(path).expanduser().resolve()
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with fmsave.open(save_path, strict=False) as career:
                info = career.info
                game_date = _save_date(info)
                table = career.players()
                players = [record_to_player(player, game_date) for player in table]
                try:
                    season_stats = _season_stats(career)
                except Exception:  # stats are optional extras; never block setup on them
                    season_stats = {}
        except (fmsave.UnsupportedGameError, fmsave.NotAFmSaveError) as exc:
            raise _unsupported(fmsave, exc, save_path) from exc
        captured = [str(item.message) for item in caught]
    unknown_build = _unknown_build_messages(fmsave, caught)
    if unknown_build and not allow_reader_warnings:
        raise UnsupportedSaveError(
            "\n".join(unknown_build)
            + f"\n{supported_saves_message()}\n"
            + "Rerun with --allow-reader-warnings to try anyway; the extracted data may be wrong."
        )
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
    checked = [p.potential_below_current for p in players if p.potential_below_current is not None]
    below = sum(checked) / len(checked) if checked else None
    if below is not None and below > MAX_PA_BELOW_CURRENT and not allow_reader_warnings:
        raise RuntimeError(
            f"Potential ability looks misread: {below:.1%} of players have potential below "
            "current ability, which the game never allows. Models trained on these labels would "
            "be unreliable. Rerun with --allow-reader-warnings to continue anyway."
        )
    return ExtractionResult(
        players=players,
        save_date=game_date,
        game=str(getattr(info, "game", "FM26")),
        build=str(getattr(info, "build", "unknown")),
        warnings=captured,
        pa_below_current_fraction=below,
        season_stats=season_stats,
    )


def read_season_stats(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read only this season's player stats (for saves that were set up before stats existed)."""
    fmsave = _import_fmsave()
    save_path = Path(path).expanduser().resolve()
    try:
        with fmsave.open(save_path, strict=False) as career:
            return _season_stats(career)
    except (fmsave.UnsupportedGameError, fmsave.NotAFmSaveError) as exc:
        raise _unsupported(fmsave, exc, save_path) from exc
