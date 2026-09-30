"""The guided experience for a game player: keys, save, setup and chat, with no configuration.

Everything a user has to do is answered in plain language here. All input and output goes
through `Console`, so the whole flow can be tested and reused without a terminal.
"""

from __future__ import annotations

import getpass
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import Settings, load_settings
from .extract import UnsupportedSaveError
from .keys import save_key
from .prepare import prepare, setup_problem
from .validate import apply_to_config, calibrate, parse_amount, spotcheck_sample
from .visible_db import VisibleStore

WELCOME = """
Football Manager 26 scouting assistant
Ask for players in plain words, for example:
  "five young central midfielders under €8M"
  "the best goalkeeper prospects you can find"
"""

KEY_HELP = {
    "DEEPSEEK_API_KEY": (
        "DeepSeek key",
        "the AI that understands your questions",
        "https://platform.deepseek.com/api_keys",
    ),
    "TABPFN_TOKEN": (
        "TabPFN key",
        "estimates how good each player can become",
        "https://platform.priorlabs.ai/account/api-keys",
    ),
}
QUIT_WORDS = {"exit", "quit", "q", "bye"}


def default_save_folders() -> list[Path]:
    home = Path.home()
    return [
        home / "Library/Application Support/Sports Interactive/Football Manager 2026/games",
        home / "Documents/Sports Interactive/Football Manager 2026/games",
        home / ".local/share/Sports Interactive/Football Manager 2026/games",
    ]


def find_saves(folder: Path, recursive: bool = False) -> list[Path]:
    """Save files (.fm) in a folder, most recently changed first."""
    if not folder.is_dir():
        return []
    files = folder.rglob("*.fm") if recursive else folder.glob("*.fm")
    return sorted((f for f in files if f.is_file()), key=lambda f: f.stat().st_mtime, reverse=True)


@dataclass
class Console:
    ask: Callable[[str], str] = input
    secret: Callable[[str], str] = getpass.getpass
    say: Callable[[str], None] = print
    dimmed: list[str] = field(default_factory=list)  # progress lines already shown


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:,.0f} MB"


def _when(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y %H:%M")


def ensure_keys(
    settings: Settings,
    console: Console,
    interactive: bool,
    only: tuple[str, ...] | None = None,
) -> None:
    """Make sure the keys are available (both, or just `only`), asking once if not."""
    for name, (label, purpose, url) in KEY_HELP.items():
        if only is not None and name not in only:
            continue
        present = settings.deepseek_api_key if name == "DEEPSEEK_API_KEY" else settings.tabpfn_ready
        if present:
            continue
        if not interactive:
            raise ValueError(f"The {label} is missing. Run fm26-agent without options to add it.")
        console.say(f"\nYou need a {label}: {purpose}.\nGet one at {url}")
        while True:
            value = console.secret(f"Paste your {label} (input is hidden): ").strip()
            try:
                save_key(settings.project_root, name, value)
                break
            except ValueError as exc:
                console.say(str(exc))
        console.say("Saved in the .env file in this folder. Keep that file private.")


def _copy_in(source: Path, settings: Settings, console: Console) -> Path:
    target = settings.project_root / source.name
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return target
    console.say(f"Copying {source.name} ({_size(source)}) into this folder...")
    shutil.copy2(source, target)
    return target


def _pick(console: Console, title: str, options: list[Path]) -> Path | None:
    console.say(title)
    for number, path in enumerate(options, 1):
        console.say(f"  {number}. {path.name}  ({_size(path)}, {_when(path)})")
    while True:
        answer = console.ask("Number (or Enter to cancel): ").strip()
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        console.say("Please type one of the numbers above.")


def choose_save(settings: Settings, console: Console, requested: Path | None) -> Path | None:
    """Find the save to work with: one that is already set up, a file here, or one from the game."""
    root = settings.project_root
    if requested is not None:
        path = requested.expanduser()
        if not path.is_file():
            raise ValueError(f"I can't find the file {path}")
        return path if path.resolve().is_relative_to(root) else _copy_in(path, settings, console)
    here = find_saves(root)
    for candidate in here:
        if setup_problem(settings, candidate.resolve()) is None:
            return candidate.resolve()
    if len(here) == 1:
        console.say(f"Found your save: {here[0].name} ({_size(here[0])}).")
        return here[0].resolve()
    if here:
        picked = _pick(console, "Which save do you want to use?", here)
        return picked.resolve() if picked else None
    from_game = [save for folder in default_save_folders() for save in find_saves(folder, True)]
    if from_game:
        picked = _pick(console, "I found these saves from the game:", from_game[:9])
        return _copy_in(picked, settings, console).resolve() if picked else None
    console.say(
        "I couldn't find a save file yet. Drag your .fm save file into this window and press "
        "Enter (or type its path)."
    )
    answer = console.ask("Save file (or Enter to cancel): ").strip().strip("'\"")
    if not answer:
        return None
    return choose_save(settings, console, Path(answer))


def offer_calibration(settings: Settings, console: Console) -> Settings:
    """Optional: match euro prices to the game using two players' in-game values."""
    store = VisibleStore(settings.data.visible_database)
    from .private_db import PrivateStore

    sample = spotcheck_sample(store, PrivateStore(settings.data.private_database), count=3)
    if len(sample) < 2:
        return settings
    console.say(
        "\nOptional: make the euro prices match your game. Look these players up in the game and "
        "type the value shown (like 4.5M). Press Enter to skip."
    )
    observed = []
    for row in sample[:3]:
        answer = console.ask(
            f"  {row['name']} ({row['club'] or 'no club'}, {row['age']}): "
        ).strip()
        if not answer:
            console.say("Skipped. You can do this later with: fm26-agent calibrate")
            return settings
        try:
            observed.append((row["name"], parse_amount(answer)))
        except ValueError as exc:
            console.say(f"{exc} Skipping this step.")
            return settings
    result = calibrate(store, observed)
    if not result["consistent"]:
        console.say("Those values don't agree with each other, so I left prices as they are.")
        return settings
    apply_to_config(settings.config_path, result["eur_per_internal_unit"])
    console.say("Prices now match your game.")
    return load_settings(settings.config_path)


_PROGRESS = (
    ("search_players", "Searching your save..."),
    ("scoring ", None),
    ("get_player_details", "Taking a closer look at the best candidates..."),
    ("shortlist ranking rejected", "Double-checking the ranking..."),
    ("running objective", "Running the objective: TabPFN is estimating every matching player..."),
)


def progress_message(message: str) -> str | None:
    """Turn the agent's technical progress lines into plain language (or nothing)."""
    if message.startswith("scoring "):
        players, _, label = message.removeprefix("scoring ").strip().partition(" for ")
        return f"Estimating {label or 'potential'} for {players}..."
    if message.startswith("building task "):
        from .targets import TARGETS

        target = TARGETS.get(message.removeprefix("building task ").strip())
        label = target.label if target else "it"
        return f"Teaching TabPFN to predict {label} and checking it on players it hasn't seen..."
    for prefix, text in _PROGRESS:
        if message.startswith(prefix):
            return text
    return None


def run(
    settings: Settings,
    console: Console | None = None,
    *,
    query: str | None = None,
    save: Path | None = None,
    allow_reader_warnings: bool = False,
    demo: bool = False,
) -> int:
    from .agent import render_shortlist
    from .runtime import open_lab, open_runtime, scout

    console = console or Console()
    interactive = query is None
    try:
        console.say(WELCOME)
        ensure_keys(settings, console, interactive)
        source = choose_save(settings, console, save)
        if source is None:
            console.say("Nothing to do without a save file. Bye!")
            return 1
        fresh = setup_problem(settings, source) is not None
        prepare(settings, source, allow_reader_warnings=allow_reader_warnings, emit=console.say)
        if fresh and interactive and not demo and not settings.currency_calibrated:
            settings = offer_calibration(settings, console)
        backend, store, predictor = open_runtime(settings)
        lab = open_lab(settings, store)
    except UnsupportedSaveError as exc:
        console.say(f"\nThis save can't be used.\n{exc}")
        return 1
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        console.say(f"\nSomething needs attention: {exc}")
        return 1
    except Exception as exc:
        console.say(
            f"\nSomething went wrong ({type(exc).__name__}). Check your internet connection and "
            "keys, then run it again; finished steps are not repeated."
        )
        return 1
    if demo:
        console.say("Demo mode: sample players (made-up names), not a real save.")
    elif not settings.currency_calibrated:
        console.say(
            "Tip: prices are approximate until they're matched to your game "
            "(run: fm26-agent calibrate)."
        )
    last: list[str] = []

    def show(message: str) -> None:
        text = progress_message(message)
        if text and text not in last[-1:]:
            last.append(text)
            console.say("  " + text)

    if lab is not None:
        return _plan_loop(settings, console, backend, store, lab, query, show, last, demo)
    while True:
        try:
            current = query if query is not None else console.ask("\nWhat are you looking for? ")
        except (EOFError, KeyboardInterrupt):
            console.say("")
            return 0
        current = current.strip()
        if current.lower() in QUIT_WORDS:
            return 0
        if not current:
            continue
        last.clear()
        result, _ = scout(settings, backend, store, predictor, current, trace=show, lab=lab)
        console.say("\n" + render_shortlist(result))
        if query is not None:
            return 1 if result.error else 0


PROMPTS = {
    "idle": "\nWhat are you looking for? ",
    "asking": "\nYour answer: ",
    "proposed": "\nGo, or change something? ",
}


def _plan_loop(settings, console, backend, store, lab, query, show, last, demo=False) -> int:
    """Planning mode: questions and an objective card first, results after "go"."""
    from .runtime import open_session, write_report, write_session_report

    session = open_session(settings, backend, store, lab, auto=query is not None, progress=show)
    while True:
        try:
            current = query if query is not None else console.ask(PROMPTS[session.state])
        except (EOFError, KeyboardInterrupt):
            console.say("")
            return 0
        current = current.strip()
        if current.lower() in QUIT_WORDS:
            return 0
        if not current and session.state != "proposed":
            continue
        last.clear()
        try:
            reply = session.send(current or "go")
        except Exception as exc:  # keep the conversation alive; details go to the report
            session.log.append({"error": f"{type(exc).__name__}: {exc}"})
            reply = None
        console.say(
            "\n"
            + (
                reply.text
                if reply is not None
                else "Sorry, something went wrong there. Try again or word it differently."
            )
        )
        if reply is not None and reply.kind == "shortlist":
            write_session_report(settings, session)
            saved = write_report(settings, "objective", reply.objective.to_dict())
            console.say(
                f"\nSaved this objective. Rerun it any time without the chat:\n"
                f"  fm26-agent find{' --demo' if demo else ''} --objective "
                f"{saved.relative_to(settings.project_root)}"
            )
        if query is not None:
            return 0 if reply is not None and reply.kind in ("shortlist", "chat") else 1


def main_interactive(settings: Settings, **kwargs) -> int:  # pragma: no cover - thin wrapper
    try:
        return run(settings, **kwargs)
    except KeyboardInterrupt:
        print("\nBye!", file=sys.stderr)
        return 130
