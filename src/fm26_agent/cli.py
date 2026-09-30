from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .config import Settings, ensure_inside, load_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FM26 scouting with DeepSeek and hosted TabPFN")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default="config.toml", help="Configuration TOML path")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check local setup and save compatibility")
    doctor.add_argument("--save", type=Path)
    doctor.add_argument("--allow-reader-warnings", action="store_true")
    prepare = commands.add_parser("prepare", help="Extract, split, fit hosted TabPFN and evaluate")
    prepare.add_argument("--save", type=Path, required=True)
    prepare.add_argument("--allow-reader-warnings", action="store_true")
    prepare.add_argument(
        "--refit",
        action="store_true",
        help="Explicitly replace the existing fixed reference and hosted fit",
    )
    prepare.add_argument(
        "--preview",
        action="store_true",
        help="Inspect sampling distributions without changing databases or calling APIs",
    )
    prepare.add_argument(
        "--extract-only",
        action="store_true",
        help="Build local databases without contacting Prior Labs",
    )
    chat = commands.add_parser("chat", help="Run the interactive scouting assistant")
    chat.add_argument("--query", help="Run one query and exit")
    prediction_mode = chat.add_mutually_exclusive_group()
    prediction_mode.add_argument(
        "--agent-only", action="store_true", help="Disable the TabPFN tool"
    )
    prediction_mode.add_argument(
        "--classifier",
        action="store_true",
        help="Rank by wonderkid probability instead of the default predicted potential",
    )
    chat.add_argument(
        "--known-values-only",
        action="store_true",
        help="Drop players whose market value the save does not store when a value filter is set "
        "(by default they are kept and flagged as value unknown)",
    )
    chat.add_argument(
        "--held-out",
        action="store_true",
        help="Restrict this query to held-out evaluation candidates instead of full-save demo",
    )
    evaluation = commands.add_parser("evaluate", help="Run the five fixed agent comparison queries")
    evaluation.add_argument(
        "--classifier",
        action="store_true",
        help="Evaluate the wonderkid-probability model instead of the default predicted potential",
    )
    spotcheck = commands.add_parser(
        "spotcheck",
        help="List players to look up in the game: compare their value and PA with the save",
    )
    spotcheck.add_argument("--count", type=int, default=8)
    calibration = commands.add_parser(
        "calibrate",
        help="Compute the euro multiplier from in-game values, e.g. calibrate 'Name=4.5M' 'Other=12M'",
    )
    calibration.add_argument("observations", nargs="+", metavar="NAME=VALUE")
    calibration.add_argument(
        "--apply",
        action="store_true",
        help="Write the result to config.toml (only when the players agree); no refit is needed",
    )
    regression_fit = commands.add_parser(
        "fit-regression",
        help="Fit/reuse the exact-PA regressor chat uses by default (uploads exact PA to Prior Labs) on the existing fixed reference; does not replace the classifier",
    )
    regression_fit.add_argument(
        "--refit", action="store_true", help="Explicitly replace the regression fit only"
    )
    commands.add_parser(
        "compare-models",
        help="Compare classifier and regressor on identical held-out pools; print actual PA/hits offline, without DeepSeek",
    )
    return parser


def _check_regression_reference(settings: Settings, preparation_id: str) -> int:
    """chat uses the regressor by default, so doctor checks its saved fit (locally, no API call)."""
    path = settings.data.regression_reference
    if path is None or not path.exists():
        print("Regression model: not fitted; run fit-regression (chat uses it by default).")
        return 1
    from .features import FeatureSchema

    payload = json.loads(path.read_text())
    fingerprint = FeatureSchema.load(settings.data.feature_schema).fingerprint
    if (
        payload.get("task") != "pa_regression"
        or payload.get("preparation_id") != preparation_id
        or payload.get("feature_fingerprint") != fingerprint
    ):
        print("Regression model: incompatible with this preparation; run fit-regression.")
        return 1
    print("Regression model: compatible fixed TabPFN 3.5 reference.")
    return 0


def doctor(config_path: str, save: Path | None, allow_reader_warnings: bool = False) -> int:
    import os

    print(f"Python: {sys.version.split()[0]} (requires 3.12+)")
    from .extract import supported_saves_message

    print("Supported saves: " + supported_saves_message())
    errors = 0
    if sys.version_info < (3, 12):  # noqa: UP036 - doctor intentionally checks the running interpreter
        errors += 1
    for package in ("fmsave", "tabpfn-client", "openai", "scikit-learn"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: missing")
            errors += 1
    llm_key_present = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY"))
    tabpfn_key_present = bool(os.getenv("TABPFN_TOKEN"))
    print("LLM API key: " + ("set" if llm_key_present else "missing"))
    print("TABPFN_TOKEN: " + ("set" if tabpfn_key_present else "missing"))
    errors += int(not llm_key_present) + int(not tabpfn_key_present)
    settings: Settings | None = None
    try:
        settings = load_settings(config_path)
        print(
            f"Currency multiplier: {settings.eur_per_internal_unit} EUR per internal unit (verify against the game)"
        )
        if not settings.currency_calibrated:
            print(
                "Currency calibration: unverified; euro values and budget filters are provisional."
            )
        print(f"LLM: {settings.llm.model} at {settings.llm.base_url}")
        if settings.data.visible_database.exists():
            from .visible_db import VisibleStore

            summary = VisibleStore(settings.data.visible_database).summary()
            print("Database: " + json.dumps(summary, indent=2))
            from .features import FEATURE_SCHEMA_VERSION

            if (
                summary.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
                or summary.get("model_version") != "v3.5"
            ):
                print(
                    "Model compatibility: outdated feature preparation. Run `fm26-agent prepare --save <your .fm save>` once for the 59-feature model; the fixed reference sampler is unchanged."
                )
                errors += 1
            elif not summary["model_ready"]:
                print("Model status: not fitted; run prepare without --extract-only.")
                errors += 1
            else:
                from .features import FeatureSchema

                schema = FeatureSchema.load(settings.data.feature_schema)
                reference = json.loads(settings.data.model_reference.read_text())
                if (
                    reference.get("version") != 2
                    or reference.get("feature_fingerprint") != schema.fingerprint
                    or reference.get("preparation_id")
                    != VisibleStore(settings.data.visible_database).metadata().get("preparation_id")
                ):
                    print(
                        "Model status: saved model, feature schema or dataset identity differs; prepare again."
                    )
                    errors += 1
                else:
                    print(
                        "Model status: compatible fixed TabPFN 3.5 reference; query filters do not refit it."
                    )
                    errors += _check_regression_reference(settings, reference["preparation_id"])
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration: {exc}")
        errors += 1
    if save:
        from .extract import read_save
        from .schema import VISIBLE_ATTRIBUTES

        print("Checking the save's player reader...")
        try:
            if settings:
                save = ensure_inside(settings.project_root, save, "--save")
            extracted = read_save(save, allow_reader_warnings)
            print(
                f"Save: {extracted.game}, build {extracted.build}, date {extracted.save_date}; {len(extracted.players):,} player records"
            )
            for warning in extracted.warnings:
                print("Reader warning: " + warning)
            for column in ("age", "name", "value_eur", "club") + VISIBLE_ATTRIBUTES:
                count = sum(player.visible.get(column) is not None for player in extracted.players)
                print(f"  {column}: {count / len(extracted.players):.1%} coverage")
        except Exception as exc:
            print(f"Save check failed: {exc}")
            errors += 1
    return 1 if errors else 0


def chat(
    settings: Settings,
    query: str | None,
    agent_only: bool,
    heldout_only: bool = False,
    regression: bool = True,
    known_values_only: bool = False,
) -> int:
    from .agent import render_shortlist
    from .runtime import open_runtime, scout

    backend, store, predictor = open_runtime(settings, not agent_only, regression)
    if agent_only:
        print("Prediction mode: none (agent judgment only).")
    elif regression:
        print(
            "Prediction mode: predicted potential (regression); estimates are not probabilities or actual hidden ability."
        )
    else:
        print("Prediction mode: wonderkid probability (classifier).")
    if not settings.currency_calibrated:
        print(
            f"Currency: uncalibrated multiplier {settings.eur_per_internal_unit}; euro values/budget filters are provisional."
        )
    print(
        "Value filters: players with no market value stored in the save are dropped."
        if known_values_only
        else "Value filters: players with no market value stored in the save are kept and flagged "
        "'value unknown'."
    )
    print(
        "Candidate scope: held-out evaluation."
        if heldout_only
        else "Candidate scope: full-save demo; training overlap is labelled, not held-out evaluation."
    )
    if query is None:
        print("FM26 scouting assistant. Each request starts fresh; type exit to quit.")
    while True:
        try:
            current = query if query is not None else input("Scout> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if current.lower() in ("exit", "quit"):
            return 0
        if not current:
            continue
        result, _ = scout(
            settings,
            backend,
            store,
            predictor,
            current,
            heldout_only=heldout_only,
            include_unknown_value=not known_values_only,
            trace=lambda message: print("  → " + message),
        )
        print(render_shortlist(result))
        if query is not None:
            return 1 if result.error else 0


def spotcheck(settings: Settings, count: int) -> int:
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
        print("Written to config.toml and marked calibrated. No refit or new prepare is needed.")
    else:
        print("Add --apply to write it to config.toml (no refit or new prepare is needed).")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args.config, args.save, args.allow_reader_warnings)
        settings = load_settings(args.config)
        if args.command == "prepare":
            from .prepare import prepare

            prepare(
                settings,
                args.save,
                allow_reader_warnings=args.allow_reader_warnings,
                extract_only=args.extract_only,
                refit=args.refit,
                preview=args.preview,
            )
            return 0
        if args.command == "chat":
            return chat(
                settings,
                args.query,
                args.agent_only,
                args.held_out,
                not args.classifier,
                args.known_values_only,
            )
        if args.command == "evaluate":
            from .evaluate import evaluate
            from .runtime import open_runtime

            backend, store, predictor = open_runtime(settings, True, not args.classifier)
            report = evaluate(settings, backend, store, predictor)
            return 0 if all(count == 5 for count in report["successful_runs"].values()) else 1
        if args.command == "spotcheck":
            return spotcheck(settings, args.count)
        if args.command == "calibrate":
            return calibrate_command(settings, args.observations, args.apply)
        if args.command == "fit-regression":
            from .regression import fit_regression

            report = fit_regression(settings, refit=args.refit)
            return 1 if report.get("model_metrics_error") else 0
        if args.command == "compare-models":
            from .regression import compare_models
            from .runtime import load_predictor
            from .visible_db import VisibleStore

            store = VisibleStore(settings.data.visible_database)
            report = compare_models(
                settings,
                load_predictor(settings, store, regression=False),
                load_predictor(settings, store, regression=True),
            )
            return (
                1
                if any(
                    "error" in model
                    for case in report["cases"]
                    for model in case["models"].values()
                )
                else 0
            )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)):
            print(f"Error: {exc}", file=sys.stderr)
        else:
            print(
                f"Error: {type(exc).__name__}. Check service credentials, quota, network connectivity and doctor diagnostics.",
                file=sys.stderr,
            )
        return 1
    return 0
