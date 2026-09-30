from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .config import Settings, ensure_inside, load_settings
from .keys import load_env_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Football Manager 26 scouting assistant. Run it without options to get started."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default="config.toml", help=argparse.SUPPRESS)
    parser.add_argument("--query", help="Ask one question and exit")
    parser.add_argument("--save", type=Path, help="Use this save file instead of looking for one")
    parser.add_argument("--allow-reader-warnings", action="store_true", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", title="other commands")
    doctor = commands.add_parser("doctor", help="Check that everything is in place")
    doctor.add_argument("--save", dest="doctor_save", type=Path, help="Also check this save file")
    commands.add_parser("spotcheck", help="List players to look up in the game")
    calibration = commands.add_parser(
        "calibrate", help="Match euro prices to the game, e.g. calibrate 'Name=4.5M' 'Other=12M'"
    )
    calibration.add_argument("observations", nargs="+", metavar="NAME=VALUE")
    calibration.add_argument("--apply", action="store_true", help="Save the result")
    prepare = commands.add_parser("prepare", help="Set up a save without starting a chat")
    prepare.add_argument("--save", dest="prepare_save", type=Path, required=True)
    prepare.add_argument("--refit", action="store_true", help="Redo the setup from scratch")
    return parser


def doctor(settings: Settings, save: Path | None = None) -> int:
    from .extract import read_save, supported_saves_message
    from .prepare import setup_problem

    problems = 0
    print(f"Python {sys.version.split()[0]} (needs 3.12 or newer)")
    if sys.version_info < (3, 12):  # noqa: UP036 - doctor intentionally checks the running interpreter
        problems += 1
    for package in ("fmsave", "tabpfn-client", "openai"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: missing")
            problems += 1
    for label, key in (
        ("DeepSeek key", settings.deepseek_api_key),
        ("TabPFN key", settings.tabpfn_token),
    ):
        print(f"{label}: {'found' if key else 'missing (you will be asked for it)'}")
    print(supported_saves_message())
    reason = setup_problem(settings)
    print("Setup: ready" if reason is None else f"Setup: not ready ({reason})")
    if save:
        try:
            extracted = read_save(
                ensure_inside(settings.project_root, save, "--save"),
            )
            print(
                f"Save: {extracted.game} build {extracted.build}, {len(extracted.players):,} players, "
                "readable"
            )
        except Exception as exc:
            print(f"Save: cannot be used. {exc}")
            problems += 1
    return 1 if problems else 0


def spotcheck(settings: Settings, count: int = 8) -> int:
    from .private_db import PrivateStore
    from .validate import spotcheck_sample
    from .visible_db import VisibleStore

    rows = spotcheck_sample(
        VisibleStore(settings.data.visible_database),
        PrivateStore(settings.data.private_database),
        count,
    )
    print("Look these players up in the game and compare (names are unique in the save):\n")
    print(f"{'Player':<28}{'Age':>4}  {'Club':<30}{'Stored value':>14}{'Extracted PA':>14}")
    for row in rows:
        print(
            f"{row['name'][:27]:<28}{row['age']:>4}  {(row['club'] or '-')[:29]:<30}"
            f"{row['stored_value']:>14,.0f}{row['extracted_pa']:>14}"
        )
    print(
        "\nValue: read each player's value in the game, then run\n"
        "  fm26-agent calibrate 'Name=4.5M' 'Other Name=12M' ...\n"
        "PA: the game shows stars only, so check exact potential in an editor "
        "(the 'Extracted PA' column should match)."
    )
    return 0


def calibrate_command(settings: Settings, observations: list[str], apply: bool) -> int:
    from .validate import apply_to_config, calibrate, parse_amount
    from .visible_db import VisibleStore

    parsed = []
    for item in observations:
        name, separator, amount = item.rpartition("=")
        if not separator or not name.strip():
            raise ValueError(f"Use NAME=VALUE, for example 'Name=4.5M', not {item!r}")
        parsed.append((name, parse_amount(amount)))
    result = calibrate(VisibleStore(settings.data.visible_database), parsed)
    for row in result["players"]:
        print(
            f"{row['name']}: stored {row['stored']:,.0f}, in game {row['in_game']:,.0f}, "
            f"ratio {row['ratio']:.4f}"
        )
    print(
        f"\nSuggested eur_per_internal_unit = {result['eur_per_internal_unit']} "
        f"(players differ by {result['spread']:.1%})"
    )
    if not result["consistent"]:
        print(
            "Not applied: use at least two players whose ratios agree within 5%. Differences "
            "usually mean a misread value or a player in another currency."
        )
        return 1
    if apply:
        apply_to_config(settings.config_path, result["eur_per_internal_unit"])
        print("Saved. Prices now match your game; nothing needs to be redone.")
    else:
        print("Add --apply to save it; nothing needs to be redone afterwards.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        load_env_file(settings.project_root)
        if args.command == "doctor":
            return doctor(settings, args.doctor_save)
        if args.command == "spotcheck":
            return spotcheck(settings)
        if args.command == "calibrate":
            return calibrate_command(settings, args.observations, args.apply)
        if args.command == "prepare":
            from .prepare import prepare

            if not settings.tabpfn_token:
                raise ValueError("Set TABPFN_TOKEN, or run fm26-agent without options to enter it")
            prepare(
                settings,
                args.prepare_save,
                allow_reader_warnings=args.allow_reader_warnings,
                refit=args.refit,
            )
            return 0
        from .app import run

        return run(
            settings,
            query=args.query,
            save=args.save,
            allow_reader_warnings=args.allow_reader_warnings,
        )
    except KeyboardInterrupt:
        print("\nBye!", file=sys.stderr)
        return 130
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)):
            print(f"Error: {exc}", file=sys.stderr)
        else:
            print(
                f"Error: {type(exc).__name__}. Check your internet connection and keys.",
                file=sys.stderr,
            )
        return 1
