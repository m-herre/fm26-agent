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
        "--regression",
        action="store_true",
        help="Rank by the separately fitted potential regressor instead of wonderkid probability",
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
    evaluation.add_argument("--regression", action="store_true")
    regression_fit = commands.add_parser(
        "fit-regression",
        help="Fit/reuse an authorized exact-PA regressor on the existing fixed reference; does not replace the classifier",
    )
    regression_fit.add_argument(
        "--refit", action="store_true", help="Explicitly replace the regression fit only"
    )
    commands.add_parser(
        "compare-models",
        help="Compare classifier and regressor on identical held-out pools; print actual PA/hits offline, without DeepSeek",
    )
    return parser


def doctor(config_path: str, save: Path | None, allow_reader_warnings: bool = False) -> int:
    import os

    print(f"Python: {sys.version.split()[0]} (requires 3.12+)")
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
            extracted = read_save(
                save, settings.eur_per_internal_unit if settings else 1.0, allow_reader_warnings
            )
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
    regression: bool = False,
    known_values_only: bool = False,
) -> int:
    from .agent import render_shortlist
    from .runtime import open_runtime, scout

    backend, store, predictor = open_runtime(settings, not agent_only, regression)
    if regression:
        print(
            "Prediction mode: potential regression; estimates are not probabilities or actual hidden ability."
        )
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
                args.regression,
                args.known_values_only,
            )
        if args.command == "evaluate":
            from .evaluate import evaluate
            from .runtime import open_runtime

            backend, store, predictor = open_runtime(settings, True, args.regression)
            report = evaluate(settings, backend, store, predictor)
            return 0 if all(count == 5 for count in report["successful_runs"].values()) else 1
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
                settings, load_predictor(settings, store), load_predictor(settings, store, True)
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
