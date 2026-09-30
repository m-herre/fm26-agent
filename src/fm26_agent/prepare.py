from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import Settings, ensure_inside
from .extract import read_save
from .features import (
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA_VERSION,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    FeatureSchema,
)
from .metrics import prediction_metrics
from .prediction import HostedPredictor
from .private_db import PrivateStore
from .sampling import SAMPLER_VERSION, representative_sample
from .schema import VISIBLE_ATTRIBUTES
from .split import stratified_cap
from .visible_db import VisibleStore


def preparation_signature(settings: Settings, source: Path) -> str:
    """Identity of everything a preparation depends on; a change means a new extraction and fit."""
    stat = source.stat()
    return hashlib.sha256(
        json.dumps(
            {
                "source": str(source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "model": "v3.5",
                "sampler": SAMPLER_VERSION,
                "training": asdict(settings.training),
                "currency": settings.eur_per_internal_unit,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def prepare(
    settings: Settings,
    save_path: Path,
    *,
    allow_reader_warnings: bool = False,
    extract_only: bool = False,
    refit: bool = False,
    preview: bool = False,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    source = ensure_inside(settings.project_root, save_path, "--save")
    signature = preparation_signature(settings, source)
    store = VisibleStore(settings.data.visible_database)
    if not refit and not extract_only and not preview and settings.data.visible_database.exists():
        metadata = store.metadata()
        if metadata.get("preparation_signature") == signature and metadata.get("model_ready"):
            schema = FeatureSchema.load(settings.data.feature_schema)
            reference = json.loads(settings.data.model_reference.read_text())
            if (
                reference.get("version") == 2
                and reference.get("preparation_id") == metadata.get("preparation_id")
                and reference.get("feature_fingerprint") == schema.fingerprint
            ):
                emit(
                    "Reusing the fixed prepared reference and fitted TabPFN 3.5 model; no fitting performed."
                )
                store.set_metadata("currency_calibrated", settings.currency_calibrated)
                report_path = settings.data.runs_directory / "preparation.json"
                report = json.loads(report_path.read_text())
                if report.get("model_metrics_error") and settings.tabpfn_token:
                    emit("Retrying held-out metrics against the saved fit; no refitting.")
                    predictor = HostedPredictor.load(
                        settings.data.model_reference,
                        settings.data.feature_schema,
                        metadata["preparation_id"],
                    )
                    test_labels = stratified_cap(
                        PrivateStore(settings.data.private_database).rows("test"),
                        settings.training.max_evaluation_rows,
                        settings.training.random_seed,
                    )
                    evaluation_players = store.get_players(
                        [row["player_id"] for row in test_labels], require_test=True
                    )
                    _evaluate_saved_fit(predictor, test_labels, evaluation_players, report, emit)
                    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                return report
    if not extract_only and not preview and not settings.tabpfn_token:
        raise ValueError("Set TABPFN_TOKEN before hosted fitting, or use --extract-only")
    emit("Reading player records from the save...")
    extracted = read_save(source, settings.eur_per_internal_unit, allow_reader_warnings)
    emit(
        f"Save: {extracted.game} build {extracted.build}, game date {extracted.save_date}; "
        f"{len(extracted.players):,} player records."
    )
    for message in extracted.warnings:
        emit(f"Reader warning: {message}")
    labels = [
        {
            "player_id": player.visible["player_id"],
            "potential_ability": player.potential_ability,
            "wonderkid": int(player.potential_ability >= settings.training.wonderkid_threshold)
            if player.potential_ability is not None
            else None,
        }
        for player in extracted.players
    ]
    emit(
        f"Selecting {settings.training.max_train_rows:,} fixed reference players and balancing marginal distributions..."
    )
    reference_ids, sampling_report = representative_sample(
        [player.visible for player in extracted.players],
        labels,
        settings.training.max_train_rows,
        settings.training.random_seed,
    )
    reference = set(reference_ids)
    model_features = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES),
        "groups": {
            "numeric": len(NUMERIC_FEATURES),
            "categorical": len(CATEGORICAL_FEATURES),
            "text": len(TEXT_FEATURES),
        },
        "columns": list(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES),
    }
    sampling_report["model_features"] = model_features
    for player, label in zip(extracted.players, labels, strict=True):
        split = (
            "train"
            if label["player_id"] in reference
            else "test"
            if label["wonderkid"] is not None
            else "unlabeled"
        )
        player.visible["split"] = label["split"] = split
    if preview:
        settings.data.runs_directory.mkdir(parents=True, exist_ok=True)
        output = settings.data.runs_directory / "reference-preview.json"
        output.write_text(json.dumps(sampling_report, indent=2), encoding="utf-8")
        emit(f"Reference preview: {output}; existing databases and model were not changed.")
        return sampling_report
    preparation_id = str(uuid.uuid4())
    visible = [player.visible for player in extracted.players]
    columns = ("age", "value_eur", "wage_eur", "club", "contract_end") + VISIBLE_ATTRIBUTES
    coverage = {
        column: sum(row.get(column) is not None for row in visible) / len(visible)
        for column in columns
    }
    metadata = {
        "preparation_id": preparation_id,
        "preparation_signature": signature,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": "v3.5",
        "reference_rows": len(reference_ids),
        "save_date": extracted.save_date.isoformat(),
        "game": extracted.game,
        "build": extracted.build,
        "eur_per_internal_unit": settings.eur_per_internal_unit,
        "currency_calibrated": settings.currency_calibrated,
        "field_coverage": coverage,
        "model_ready": False,
        "available_positions": sorted(
            {
                position
                for row in visible
                for position in row["natural_positions"] + row["accomplished_positions"]
            }
        ),
    }
    private = PrivateStore(settings.data.private_database)
    # Each SQLite replacement is transactional. model_ready stays false until fitting succeeds.
    store.initialize(visible, metadata)
    private.initialize(labels, preparation_id)
    report: dict[str, Any] = {
        "preparation_id": preparation_id,
        "preparation_signature": signature,
        "sampling": sampling_report,
        "model_features": model_features,
        "currency_calibrated": settings.currency_calibrated,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "warnings": extracted.warnings,
        "player_count": len(visible),
        "unknown_pa_count": sum(row["wonderkid"] is None for row in labels),
        "threshold": settings.training.wonderkid_threshold,
        "split_seed": settings.training.random_seed,
        "held_out_count": sum(row["split"] == "test" for row in labels),
        "field_coverage": coverage,
        "versions": {
            package: version(package) for package in ("fmsave", "tabpfn-client", "scikit-learn")
        },
    }
    emit(
        f"Extracted {len(visible):,} players; {report['unknown_pa_count']:,} excluded from supervised work because exact PA is unavailable."
    )
    settings.data.runs_directory.mkdir(parents=True, exist_ok=True)
    try:
        if not extract_only:
            fit_model(settings, store, private, report, emit)
    finally:
        (settings.data.runs_directory / "preparation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def fit_model(
    settings: Settings,
    store: VisibleStore,
    private: PrivateStore,
    report: dict[str, Any],
    emit: Callable[[str], None],
) -> None:
    if not settings.tabpfn_token:
        raise ValueError("Set TABPFN_TOKEN before hosted fitting, or use --extract-only")
    train_labels = private.rows("train")
    if len(train_labels) != settings.training.max_train_rows:
        raise ValueError("Prepared reference size differs from configuration; run prepare again")
    test_labels = stratified_cap(
        private.rows("test"), settings.training.max_evaluation_rows, settings.training.random_seed
    )
    training_players = store.get_players([row["player_id"] for row in train_labels])
    evaluation_players = store.get_players(
        [row["player_id"] for row in test_labels], require_test=True
    )
    schema = FeatureSchema.fit(training_players)
    quote = HostedPredictor.estimate_cost(
        schema.transform(training_players), schema.transform(evaluation_players)
    )
    emit("Hosted cost estimate: " + json.dumps(quote, default=str))
    emit(
        f"Fitting TabPFN on {len(training_players):,} reference players with {len(schema.columns)} observable features..."
    )
    predictor = HostedPredictor.fit(
        training_players,
        [row["wonderkid"] for row in train_labels],
        schema,
        settings.training.random_seed,
    )
    schema.save(settings.data.feature_schema)
    predictor.save(settings.data.model_reference, report["preparation_id"])
    report.update(
        {
            "train_rows": len(training_players),
            "evaluation_rows": len(evaluation_players),
            "train_positive_fraction": sum(row["wonderkid"] for row in train_labels)
            / len(train_labels),
            "feature_fingerprint": schema.fingerprint,
            "hosted_cost_estimate": quote,
        }
    )
    store.set_metadata("model_ready", True)
    _evaluate_saved_fit(predictor, test_labels, evaluation_players, report, emit)


def _evaluate_saved_fit(predictor, test_labels, evaluation_players, report, emit):
    try:
        predictions = predictor.predict(evaluation_players)
        report["model_metrics"] = prediction_metrics(
            predictions, {row["player_id"]: row for row in test_labels}
        )
        report.pop("model_metrics_error", None)
        emit("Held-out metrics: " + json.dumps(report["model_metrics"], indent=2))
    except Exception as exc:
        # A prediction/quota failure must never discard a successful, potentially expensive fit.
        report["model_metrics_error"] = (
            f"{type(exc).__name__}: held-out scoring failed; saved fit is retained"
        )
        emit(
            "Warning: "
            + report["model_metrics_error"]
            + ". Run prepare again to retry metrics without fitting."
        )
