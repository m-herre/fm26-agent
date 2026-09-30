from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .config import Settings, demo_settings, ensure_inside, load_settings
from .keys import load_env_file
from .tabpfn_backend import use_project_weights
from .targets import TARGETS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Football Manager 26 scouting assistant. Run it without options to get started."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default="config.toml", help=argparse.SUPPRESS)
    parser.add_argument("--query", help="Ask one question and exit")
    parser.add_argument("--save", type=Path, help="Use this save file instead of looking for one")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Try it on the included sample players instead of a save (no Football Manager needed)",
    )
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
    find = commands.add_parser(
        "find",
        help="Scout with explicit filters, no chat needed (only the TabPFN key)",
        description="Rank players by TabPFN-estimated potential using explicit filters, e.g. "
        "fm26-agent find --position MC --age-max 20 --max-value 20M",
    )
    find.add_argument(
        "--demo", dest="find_demo", action="store_true", help="use the sample players"
    )
    find.add_argument(
        "--position", help="FM position code: GK, DC, DL, DR, DM, MC, AMC, AML, AMR, STC ..."
    )
    find.add_argument("--age-min", type=int)
    find.add_argument("--age-max", type=int)
    find.add_argument("--min-value", help="e.g. 2M or 500K")
    find.add_argument("--max-value", help="e.g. 20M")
    find.add_argument("--club", action="append", help="club name (repeat for several)")
    find.add_argument("--foot", choices=("left", "right", "both"))
    find.add_argument("--contract-within", type=int, metavar="DAYS", help="contract ends within")
    find.add_argument(
        "--like", metavar="NAME_OR_ID", help="players whose profile resembles this one"
    )
    find.add_argument("--count", type=int, default=5, choices=range(1, 26), metavar="1-25")
    find.add_argument(
        "--rank",
        choices=("expected", "ceiling", "safe", "chance"),
        default="expected",
        help="expected estimate, best case (upside), worst case (safe bet) or chance of meeting "
        "the threshold",
    )
    find.add_argument(
        "--predict",
        metavar="TARGET",
        choices=[name for name in TARGETS if name != "potential_ability"],
        help="rank by another hidden value TabPFN learns on the spot: "
        + ", ".join(name for name in TARGETS if name != "potential_ability"),
    )
    find.add_argument(
        "--threshold",
        type=float,
        help="with --predict: the level that counts (at least, or at most where low is good)",
    )
    prepare = commands.add_parser("prepare", help="Set up a save without starting a chat")
    prepare.add_argument("--save", dest="prepare_save", type=Path, required=True)
    prepare.add_argument("--refit", action="store_true", help="Redo the setup from scratch")
    return parser


DEMO_SAMPLE = Path("sample") / "players.csv.gz"


def find_command(settings: Settings, args: argparse.Namespace) -> int:
    from .agent import render_shortlist
    from .app import Console, ensure_keys, progress_message
    from .custom_tasks import TaskSpec
    from .finder import find_players
    from .prepare import add_missing_estimates, prepare, setup_problem
    from .runtime import load_predictor, open_lab, write_report
    from .tools import ScoutingTools
    from .validate import parse_amount
    from .visible_db import VisibleStore

    console = Console()
    ensure_keys(settings, console, sys.stdin.isatty(), only=("TABPFN_TOKEN",))
    if args.find_demo or args.demo:
        sample = settings.project_root / DEMO_SAMPLE
        settings = demo_settings(settings)
        if setup_problem(settings, sample.resolve()) is not None:
            prepare(settings, sample, emit=console.say)
    elif (reason := setup_problem(settings)) is not None:
        raise ValueError(f"No save is ready ({reason}). Run fm26-agent once, or add --demo")
    add_missing_estimates(settings, console.say)
    if args.threshold is not None and not args.predict:
        raise ValueError("--threshold goes with --predict")
    if args.rank == "chance" and args.predict and args.threshold is None:
        raise ValueError("--rank chance with --predict needs a --threshold")
    store = VisibleStore(settings.data.visible_database)
    lab = open_lab(settings, store) if args.predict else None
    if args.predict and (lab is None or args.predict not in lab.available()):
        raise ValueError(
            "This setup has no hidden values to learn from yet. Run fm26-agent prepare --save "
            "<your save> once to add them (nothing is refitted)."
        )
    tools = ScoutingTools(
        store,
        load_predictor(settings, store),
        currency=settings.currency,
        lab=lab,
    )
    tools.progress = lambda message: (
        (text := progress_message(message)) and console.say("  " + text)
    )
    result = find_players(
        store,
        tools,
        count=args.count,
        rank_by=args.rank,
        like=args.like,
        task=TaskSpec(args.predict, args.threshold) if args.predict else None,
        position=args.position,
        age_min=args.age_min,
        age_max=args.age_max,
        value_min_eur=parse_amount(args.min_value) if args.min_value else None,
        value_max_eur=parse_amount(args.max_value) if args.max_value else None,
        club=args.club,
        preferred_foot=args.foot,
        contract_ends_within_days=args.contract_within,
    )
    write_report(settings, "find", result.to_dict())
    console.say("\n" + render_shortlist(result))
    return 0


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
        if label == "TabPFN key" and settings.tabpfn_backend == "local":
            from .tabpfn_backend import device

            print(f"TabPFN: runs on this computer ({device()}), no key needed")
            continue
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
        use_project_weights(settings.project_root)
        if args.command == "doctor":
            return doctor(settings, args.doctor_save)
        if args.command == "spotcheck":
            return spotcheck(settings)
        if args.command == "calibrate":
            return calibrate_command(settings, args.observations, args.apply)
        if args.command == "find":
            return find_command(settings, args)
        if args.command == "prepare":
            from .prepare import prepare

            if not settings.tabpfn_ready:
                raise ValueError("Set TABPFN_TOKEN, or run fm26-agent without options to enter it")
            prepare(
                settings,
                args.prepare_save,
                allow_reader_warnings=args.allow_reader_warnings,
                refit=args.refit,
            )
            return 0
        from .app import run

        save = args.save
        if args.demo:
            save = settings.project_root / "sample" / "players.csv.gz"
            if not save.exists():
                raise ValueError(
                    "The demo data is missing. Create it from a prepared save with "
                    "python scripts/export_sample.py"
                )
            settings = demo_settings(settings)
        return run(
            settings,
            query=args.query,
            save=save,
            demo=args.demo,
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
