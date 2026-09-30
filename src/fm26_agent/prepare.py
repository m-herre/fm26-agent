from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Settings, ensure_inside
from .extract import SEASON_STATS_VERSION, read_save, read_season_stats
from .features import FEATURE_SCHEMA_VERSION, FeatureSchema
from .prediction import HostedPredictor
from .private_db import PrivateStore
from .sample import read_sample
from .sampling import SAMPLER_VERSION, representative_sample
from .schema import VISIBLE_ATTRIBUTES
from .visible_db import VisibleStore


def preparation_signature(settings: Settings, source: Path) -> str:
    """Identity of everything a preparation depends on; a change means a new extraction and fit."""
    stat = source.stat()
    training = settings.training
    return hashlib.sha256(
        json.dumps(
            {
                "source": str(source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "model": "v3.5",
                "sampler": SAMPLER_VERSION,
                "wonderkid_threshold": training.wonderkid_threshold,
                "random_seed": training.random_seed,
                "max_train_rows": training.max_train_rows,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def setup_problem(settings: Settings, source: Path | None = None) -> str | None:
    """Why this project is not ready to answer questions, or None if it is. Local checks only.

    With `source`, the prepared data must also come from exactly that save file.
    """
    if not settings.data.visible_database.exists():
        return "no save has been set up yet"
    store = VisibleStore(settings.data.visible_database)
    metadata = store.metadata()
    if not metadata.get("model_ready"):
        return "setup did not finish"
    if source is not None and metadata.get("preparation_signature") != preparation_signature(
        settings, source
    ):
        return "the save file has changed since setup"
    try:
        schema = FeatureSchema.load(settings.data.feature_schema)
    except (FileNotFoundError, ValueError):
        return "the feature schema is missing or outdated"
    return HostedPredictor.check_reference(
        settings.data.model_reference, schema, metadata.get("preparation_id", "")
    )


def prepare(
    settings: Settings,
    save_path: Path,
    *,
    allow_reader_warnings: bool = False,
    refit: bool = False,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Read the save, pick the training players, fit the potential model once and store it all.

    Safe to call again: if this exact save was already set up, nothing is read or fitted.
    """
    source = ensure_inside(settings.project_root, save_path, "--save")
    if not refit and setup_problem(settings, source) is None:
        store = VisibleStore(settings.data.visible_database)
        if (
            source.suffix != ".gz"
            and store.metadata().get("season_stats_version") != SEASON_STATS_VERSION
        ):
            # Set up before stats existed: add them without reading players again or refitting.
            emit("Adding this season's player stats...")
            try:
                store.set_season_stats(read_season_stats(source), SEASON_STATS_VERSION)
            except Exception:
                store.set_season_stats({}, SEASON_STATS_VERSION)
                emit("Note: season stats could not be read from this save, so they won't be shown.")
        else:
            emit("This save is already set up.")
        return store.metadata()
    if not settings.tabpfn_token:
        raise ValueError("A TabPFN key is needed to set up a save")
    signature = preparation_signature(settings, source)
    emit("Reading your save...")
    extracted = (
        read_sample(source) if source.suffix == ".gz" else read_save(source, allow_reader_warnings)
    )
    emit(
        f"Found {len(extracted.players):,} players "
        f"({extracted.game} build {extracted.build}, game date {extracted.save_date})."
    )
    for message in extracted.warnings:
        emit(f"Note: {message}")
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
    emit(f"Choosing {settings.training.max_train_rows:,} representative players to learn from...")
    reference_ids, _ = representative_sample(
        [player.visible for player in extracted.players],
        labels,
        settings.training.max_train_rows,
        settings.training.random_seed,
    )
    reference = set(reference_ids)
    for player, label in zip(extracted.players, labels, strict=True):
        split = (
            "train"
            if label["player_id"] in reference
            else "test"
            if label["wonderkid"] is not None
            else "unlabeled"
        )
        player.visible["split"] = label["split"] = split
    visible = [player.visible for player in extracted.players]
    columns = ("age", "value_eur", "wage_eur", "club", "contract_end") + VISIBLE_ATTRIBUTES
    preparation_id = str(uuid.uuid4())
    metadata = {
        "preparation_id": preparation_id,
        "preparation_signature": signature,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": "v3.5",
        "source": source.name,
        "reference_rows": len(reference_ids),
        "save_date": extracted.save_date.isoformat(),
        "game": extracted.game,
        "build": extracted.build,
        "field_coverage": {
            column: sum(row.get(column) is not None for row in visible) / len(visible)
            for column in columns
        },
        "season_stats_version": SEASON_STATS_VERSION,
        "model_ready": False,
        "available_positions": sorted(
            {
                position
                for row in visible
                for position in row["natural_positions"] + row["accomplished_positions"]
            }
        ),
    }
    store = VisibleStore(settings.data.visible_database)
    private = PrivateStore(settings.data.private_database)
    # model_ready stays false until fitting succeeds, so a failed setup is never mistaken for done.
    store.initialize(visible, metadata, extracted.season_stats)
    private.initialize(labels, preparation_id)
    _fit(settings, store, private, emit)
    return store.metadata()


def _fit(
    settings: Settings, store: VisibleStore, private: PrivateStore, emit: Callable[[str], None]
) -> None:
    training = private.rows("train")
    if len(training) != settings.training.max_train_rows:
        raise ValueError("Prepared reference size differs from configuration; run prepare again")
    players = store.get_players([row["player_id"] for row in training])
    schema = FeatureSchema.fit(players)
    emit("Teaching the potential model. This can take a minute or two the first time...")
    predictor = HostedPredictor.fit(
        players,
        [row["potential_ability"] for row in training],
        schema,
        settings.training.random_seed,
    )
    schema.save(settings.data.feature_schema)
    predictor.save(settings.data.model_reference, store.metadata()["preparation_id"])
    store.set_metadata("model_ready", True)
    emit("All set.")
