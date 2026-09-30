"""Authorized regression training and private, model-only evaluation.

This module is not imported by scouting tools or the agent. The target is
uploaded only in fit(); actual labels appear only in these offline reports.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from time import perf_counter

from .evaluate import BENCHMARKS, benchmark_constraints, benchmark_pool
from .features import FeatureSchema
from .metrics import prediction_metrics, regression_metrics
from .prediction import HostedRegressionPredictor
from .private_db import PrivateStore
from .runtime import write_report
from .split import stratified_cap
from .visible_db import VisibleStore


def reference_path(settings):
    return settings.data.regression_reference or settings.data.model_reference.with_name(
        "regression-model.json"
    )


def _snapshot(settings):
    store = VisibleStore(settings.data.visible_database)
    private = PrivateStore(settings.data.private_database)
    metadata = store.metadata()
    if not metadata.get("model_ready"):
        raise ValueError(
            "Prepare the classifier dataset and fit first; regression must share its ready reference"
        )
    if private.preparation_id() != metadata.get("preparation_id"):
        raise ValueError("Visible database and private labels belong to different preparations")
    if metadata.get("eur_per_internal_unit") != settings.eur_per_internal_unit:
        raise ValueError("Currency changed; prepare the classifier dataset first")
    schema = FeatureSchema.load(settings.data.feature_schema)
    if len(schema.columns) != 59:
        raise ValueError("Regression experiment requires the prepared 59-feature schema")
    preparation = json.loads((settings.data.runs_directory / "preparation.json").read_text())
    if (
        preparation.get("preparation_id") != private.preparation_id()
        or preparation.get("split_seed") != settings.training.random_seed
        or preparation.get("threshold") != settings.training.wonderkid_threshold
    ):
        raise ValueError("Preparation identity, seed or target threshold changed; prepare first")
    train = private.rows("train")
    test = private.rows("test")
    if len(train) != settings.training.max_train_rows or any(
        row["potential_ability"] is None for row in train + test
    ):
        raise ValueError("Reference size differs or exact regression targets are unavailable")
    if (
        FeatureSchema.fit(store.get_players([r["player_id"] for r in train])).fingerprint
        != schema.fingerprint
    ):
        raise ValueError("Reference players no longer match the prepared feature schema")
    return (
        store,
        private,
        schema,
        train,
        stratified_cap(test, settings.training.max_evaluation_rows, settings.training.random_seed),
    )


def _id_hash(rows):
    return hashlib.sha256(json.dumps([row["player_id"] for row in rows]).encode()).hexdigest()


def fit_regression(settings, *, refit=False, emit=print):
    store, private, schema, train, test = _snapshot(settings)
    path = reference_path(settings)
    output = settings.data.runs_directory / "regression-fit.json"
    train_players = store.get_players([r["player_id"] for r in train])
    test_players = store.get_players([r["player_id"] for r in test], require_test=True)
    reusable = False
    if path.exists() and not refit:
        try:
            predictor = HostedRegressionPredictor.load(
                path, settings.data.feature_schema, private.preparation_id()
            )
            reusable = True
        except ValueError:
            emit(
                "Existing regression reference is incompatible with this preparation; fitting a new regression reference."
            )
    if reusable:
        emit("Reusing saved PA-regression fit; no fitting performed.")
        report = json.loads(output.read_text()) if output.exists() else {}
        if report.get("preparation_id") != private.preparation_id():
            report = {}
    else:
        if not settings.tabpfn_token:
            raise ValueError("Set TABPFN_TOKEN before fitting regression")
        quote = HostedRegressionPredictor.estimate_cost(
            schema.transform(train_players), schema.transform(test_players)
        )
        emit("Hosted regression cost estimate: " + json.dumps(quote, default=str))
        emit(
            f"Fitting TabPFN 3.5 regression on the same {len(train):,} reference players and 59 features. User-authorized upload: exact PA targets; no identities or targets in X."
        )
        predictor = HostedRegressionPredictor.fit(
            train_players,
            [r["potential_ability"] for r in train],
            schema,
            settings.training.random_seed,
        )
        predictor.save(path, private.preparation_id())
        report = {"created_at": datetime.now(UTC).isoformat(), "hosted_cost_estimate": quote}
    needs_metrics = not report.get("model_metrics") or report.get("evaluation_id_hash") != _id_hash(
        test
    )
    if needs_metrics:
        report.pop("model_metrics", None)
    report.update(
        {
            "task": "pa_regression",
            "target_upload": "exact_pa_user_authorized",
            "preparation_id": private.preparation_id(),
            "feature_fingerprint": schema.fingerprint,
            "feature_count": 59,
            "train_rows": len(train),
            "evaluation_rows": len(test),
            "reference_id_hash": _id_hash(train),
            "evaluation_id_hash": _id_hash(test),
            "random_seed": settings.training.random_seed,
            "prediction_clip": [1, 200],
        }
    )
    if settings.tabpfn_token and (not reusable or needs_metrics):
        try:
            report["model_metrics"] = regression_metrics(
                predictor.predict(test_players), private.get([r["player_id"] for r in test])
            )
            report.pop("model_metrics_error", None)
            emit("Held-out regression metrics: " + json.dumps(report["model_metrics"], indent=2))
        except Exception as exc:
            report["model_metrics_error"] = (
                f"{type(exc).__name__}: held-out scoring failed; saved regression fit is retained"
            )
            emit(
                report["model_metrics_error"]
                + "; rerun fit-regression to retry metrics without refitting."
            )
    elif needs_metrics:
        report["model_metrics_error"] = (
            "Set TABPFN_TOKEN and rerun fit-regression for metrics on the current evaluation rows; saved fit is retained"
        )
    settings.data.runs_directory.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    emit(f"Regression fit report: {output}")
    return report


def compare_models(settings, classifier, regressor, *, emit=print):
    store, private, schema, train, test = _snapshot(settings)
    all_labels = private.rows("test")
    truth = private.get([r["player_id"] for r in all_labels])
    players = store.get_players([r["player_id"] for r in all_labels], require_test=True)
    cases = [
        (
            "global_held_out_sample",
            store.get_players([r["player_id"] for r in test], require_test=True),
            None,
        )
    ]
    for name, _, position in BENCHMARKS:
        cases.append((name, benchmark_pool(players, position), benchmark_constraints(position)))
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "binary-vs-exact-pa-regression-v1",
        "preparation_id": private.preparation_id(),
        "feature_fingerprint": schema.fingerprint,
        "reference_id_hash": _id_hash(train),
        "evaluation_id_hash": _id_hash(test),
        "feature_count": 59,
        "currency_calibrated": settings.currency_calibrated,
        "random_seed": settings.training.random_seed,
        "wonderkid_threshold": settings.training.wonderkid_threshold,
        "model_reference_hashes": {
            "classifier": hashlib.sha256(settings.data.model_reference.read_bytes()).hexdigest(),
            "regression": hashlib.sha256(reference_path(settings).read_bytes()).hexdigest(),
        },
        "scope": "held_out",
        "cases": [],
    }
    for name, eligible, constraints in cases:
        entry = {
            "id": name,
            "constraints": constraints,
            "eligible_count": len(eligible),
            "eligible_wonderkid_count": sum(truth[p["player_id"]]["wonderkid"] for p in eligible),
            "models": {},
        }
        emit(
            f"{name}: {len(eligible):,} held-out players, {entry['eligible_wonderkid_count']} actual wonderkids"
        )
        by_id = {p["player_id"]: p for p in eligible}
        for mode, predictor in {"classifier": classifier, "regression": regressor}.items():
            try:
                started = perf_counter()
                scores = predictor.predict(eligible)
                elapsed = perf_counter() - started
                if len(scores) != len(eligible) or {r["player_id"] for r in scores} != set(by_id):
                    raise ValueError("Model scores do not cover the identical eligible pool")
                key = getattr(predictor, "score_field", "wonderkid_probability")
                ranked = sorted(scores, key=lambda r: (-r[key], r["player_id"]))
                metrics = (
                    prediction_metrics(scores, truth)
                    if key == "wonderkid_probability"
                    else regression_metrics(scores, truth)
                )
                shortlist = [
                    {
                        "rank": rank,
                        "player_id": r["player_id"],
                        "name": by_id[r["player_id"]]["name"],
                        key: r[key],
                        "actual_pa": truth[r["player_id"]]["potential_ability"],
                        "actual_wonderkid": bool(truth[r["player_id"]]["wonderkid"]),
                    }
                    for rank, r in enumerate(ranked[:10], 1)
                ]
                wonderkid_ranks = [
                    {
                        "rank": rank,
                        "player_id": r["player_id"],
                        "name": by_id[r["player_id"]]["name"],
                        "actual_pa": truth[r["player_id"]]["potential_ability"],
                    }
                    for rank, r in enumerate(ranked, 1)
                    if truth[r["player_id"]]["wonderkid"]
                ]
                entry["models"][mode] = {
                    "metrics": metrics,
                    "shortlist": shortlist,
                    "wonderkid_ranks": wonderkid_ranks,
                    "scored_count": len(scores),
                    "complete": True,
                    "prediction_wall_seconds": elapsed,
                }
                emit(
                    f"  {mode}: top 5 hits {metrics['true_wonderkids_top_5']}/{min(5, len(scores))}; average actual PA {metrics['average_hidden_pa_top_5']}; top 10 hits {metrics['true_wonderkids_top_10']}/{min(10, len(scores))}; prediction {elapsed:.2f}s"
                )
                for row in shortlist[:5]:
                    emit(
                        f"    {row['rank']}. {row['name']}: actual PA {row['actual_pa']} — {'wonderkid' if row['actual_wonderkid'] else 'not a wonderkid'}"
                    )
                if name != "global_held_out_sample":
                    for row in wonderkid_ranks:
                        emit(
                            f"    Actual wonderkid: {row['name']}, PA {row['actual_pa']}, rank {row['rank']}"
                        )
            except Exception as exc:
                entry["models"][mode] = {
                    "error": f"{type(exc).__name__}: model comparison failed; no partial ranking accepted"
                }
                emit(f"  {mode}: {entry['models'][mode]['error']}")
        report["cases"].append(entry)
    path = write_report(settings, "model-comparison", report)
    emit(f"Private model-comparison report: {path}")
    return report
